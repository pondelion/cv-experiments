"""画像生成モデル FLUX.2 [klein] 4B(fp8 版の本体)を diffusers のパイプラインとして読み込む。

- 本体(transformer): black-forest-labs/FLUX.2-klein-4b-fp8 の 1 ファイル(4.07GB)。重みは fp8(float8_e4m3fn)で、
  線形層ごとに値を元に戻す倍率(weight_scale)が別に入っている形式。diffusers はこの倍率を扱えないので、
  読み込むときに「fp8 の値 × 倍率」で bf16 に戻してから、diffusers の名前に変換する
  (入力側の倍率 input_scale は fp8 のまま計算するときだけ使うものなので捨てる)
- 文章を読む部分(Qwen3 4B)・VAE・設定: black-forest-labs/FLUX.2-klein-4B(bf16 版)のリポジトリから
- どちらも Hugging Face の既定のキャッシュから読む(local_files_only。ネットワークには出ない)
- ライセンス: どちらも Apache 2.0

使い方:
    pipe = load_pipeline()
    image = pipe(prompt="a photo of a dog", height=1024, width=1024, num_inference_steps=4).images[0]
    (最初の位置の引数は編集用の入力画像なので、プロンプトは必ずキーワードで渡す)
"""
import torch
from diffusers import Flux2KleinInpaintPipeline, Flux2KleinPipeline, Flux2Transformer2DModel
from diffusers.loaders.single_file_utils import convert_flux2_transformer_checkpoint_to_diffusers
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

BASE_REPO = "black-forest-labs/FLUX.2-klein-4B"
FP8_REPO = "black-forest-labs/FLUX.2-klein-4b-fp8"
FP8_FILE = "flux-2-klein-4b-fp8.safetensors"
DEFAULT_STEPS = 4  # 蒸留版(step 数を減らすよう学習したもの)の既定の step 数


def dequantize_scaled_fp8(state_dict: dict, dtype: torch.dtype) -> dict:
    """倍率付きの fp8 の重みを、指定の dtype(bf16 など)の重みに戻す。

    Args:
        state_dict: fp8 版の重み。`{層}.weight`(fp8)と `{層}.weight_scale`(float32 の倍率)の組を含む。
        dtype: 戻した後の型。

    Returns:
        倍率を掛けて戻した重み(weight_scale・input_scale のキーは除く)。
    """
    out = {}
    for key, value in state_dict.items():
        if key.endswith((".weight_scale", ".input_scale")):
            continue
        scale = state_dict.get(key[: -len(".weight")] + ".weight_scale") if key.endswith(".weight") else None
        if scale is not None:
            value = value.to(torch.float32) * scale.to(torch.float32)  # fp8 の値 × 倍率 = 元の重み
        out[key] = value.to(dtype)
    return out


def load_transformer(dtype: torch.dtype = torch.bfloat16) -> Flux2Transformer2DModel:
    """fp8 版の本体を読み込み、bf16 に戻して diffusers の Flux2Transformer2DModel にする。

    Args:
        dtype: 戻す型。

    Returns:
        Flux2Transformer2DModel。
    """
    path = hf_hub_download(FP8_REPO, FP8_FILE, local_files_only=True)
    state_dict = convert_flux2_transformer_checkpoint_to_diffusers(dequantize_scaled_fp8(load_file(path), dtype))
    config = Flux2Transformer2DModel.load_config(BASE_REPO, subfolder="transformer", local_files_only=True)
    model = Flux2Transformer2DModel.from_config(config).to(dtype)
    model.load_state_dict(state_dict, strict=True)  # 名前・形が全部合わないとエラーにする
    return model


def load_pipeline(device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> Flux2KleinPipeline:
    """FLUX.2 [klein] 4B のパイプライン(本体は fp8 版を bf16 に戻したもの)を読み込む。

    Args:
        device: 置く装置。
        dtype: 重みの型。

    Returns:
        Flux2KleinPipeline。
    """
    pipe = Flux2KleinPipeline.from_pretrained(BASE_REPO, transformer=load_transformer(dtype), torch_dtype=dtype,
                                              local_files_only=True)
    return pipe.to(device)


def inpaint_pipeline(pipe: Flux2KleinPipeline) -> Flux2KleinInpaintPipeline:
    """読み込み済みのパイプラインと重みを共有した inpainting のパイプラインを作る(GPU のメモリは増えない)。

    蒸留版の設定(is_distilled)を明示して引き継ぐ(引き継がれないと、ガイダンスのために計算を 2 回分流してしまう)。
    型(torch_dtype)も明示する: 指定しないと from_pipe が共有している重みを float32 に変えてしまい、
    本体が約 8GB → 約 16GB になって VRAM があふれる(元のパイプラインの重みも float32 になる)。
    使い方: inpaint(prompt_embeds=..., image=画像, mask_image=マスク(白 = 描き直す、黒 = 残す), strength=1.0,
                    guidance_scale=1.0, num_inference_steps=DEFAULT_STEPS)。strength=1.0 なら白い部分を丸ごと描き直す

    Args:
        pipe: load_pipeline のパイプライン。

    Returns:
        重みを共有した Flux2KleinInpaintPipeline。
    """
    return Flux2KleinInpaintPipeline.from_pipe(pipe, is_distilled=pipe.config.is_distilled, torch_dtype=pipe.transformer.dtype)


@torch.inference_mode()
def encode_prompts(pipe: Flux2KleinPipeline, prompts: list[str]) -> list[torch.Tensor]:
    """プロンプトを文章の特徴(prompt_embeds)にまとめて変える。生成のときに prompt_embeds=... で渡す。

    文章を読む部分(Qwen3 4B、bf16 で約 8GB)はこのときだけ使うので、先に全部変えてから offload_text_encoder で
    CPU に退かせると、生成中の GPU のメモリを約 8GB 減らせる。

    Args:
        pipe: load_pipeline のパイプライン(文章を読む部分が GPU にあること)。
        prompts: プロンプトの並び。

    Returns:
        プロンプトごとの特徴 (1, L(文章のトークンの数 = 512), D(文章の特徴の次元)) の並び。
    """
    return [pipe.encode_prompt(prompt=p, device=pipe.transformer.device)[0] for p in prompts]


def offload_text_encoder(pipe: Flux2KleinPipeline) -> None:
    """文章を読む部分を CPU に移し、GPU のメモリを空ける(以降は prompt_embeds で生成する)。

    Args:
        pipe: load_pipeline のパイプライン。
    """
    pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()
