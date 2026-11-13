"""inpainting 専用のモデル FLUX.1 Fill [dev] を diffusers のパイプラインとして読み込む。

FLUX.2 [klein] の inpainting(Flux2KleinInpaintPipeline)は、普通の文章から画像を作るモデルの潜在の表現を毎 step 上書きするだけで、
残す部分(物体)を条件として直接は受け取らないので、物体と背景の物理的な整合(接地・水面・大きさ)が弱かった。
FLUX.1 Fill は、マスクとマスクした画像を追加の入力として受け取るよう学習された inpainting 専用のモデル。

- 本体(12B)と文章を読む部分 T5-XXL: diffusers/FLUX.1-Fill-dev-nf4(bitsandbytes の 4 bit に量子化済み。読み込みに bitsandbytes が要る)
- CLIP の文章を読む部分・VAE・トークナイザ・設定: black-forest-labs/FLUX.1-Fill-dev(利用申請が要る)
- どちらも Hugging Face の既定のキャッシュから読む(local_files_only。ネットワークには出ない)
- ライセンス: FLUX.1 [dev] の非商用ライセンス(研究用)

使い方:
    fill = load_fill_pipeline()
    image = fill(prompt=..., image=画像, mask_image=マスク(白 = 描き直す、黒 = 残す), height=768, width=768,
                 num_inference_steps=DEFAULT_STEPS, guidance_scale=DEFAULT_GUIDANCE).images[0]
"""
import torch
from diffusers import FluxFillPipeline, FluxTransformer2DModel
from transformers import T5EncoderModel

BASE_REPO = "black-forest-labs/FLUX.1-Fill-dev"
NF4_REPO = "diffusers/FLUX.1-Fill-dev-nf4"
DEFAULT_STEPS = 28  # 以前の検証(infra-memo の FLUX1fill のノートブック)と同じ
DEFAULT_GUIDANCE = 30.0  # FLUX.1 Fill の公式の例の値(ガイダンスは蒸留されていて、ネガティブプロンプトは効かない)


def load_fill_pipeline(device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> FluxFillPipeline:
    """量子化済みの本体・T5 と、元のリポジトリの残りの部品を組み合わせて、FLUX.1 Fill のパイプラインを読み込む。

    Args:
        device: 置く装置。
        dtype: 計算の型。

    Returns:
        FluxFillPipeline。
    """
    transformer = FluxTransformer2DModel.from_pretrained(NF4_REPO, subfolder="transformer", torch_dtype=dtype,
                                                        local_files_only=True)
    text_encoder_2 = T5EncoderModel.from_pretrained(NF4_REPO, subfolder="text_encoder_2", torch_dtype=dtype,
                                                    local_files_only=True)
    pipe = FluxFillPipeline.from_pretrained(BASE_REPO, transformer=transformer, text_encoder_2=text_encoder_2,
                                            torch_dtype=dtype, local_files_only=True)
    return pipe.to(device)
