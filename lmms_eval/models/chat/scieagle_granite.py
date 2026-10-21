"""SciEagle-Granite-VL (standalone, trust_remote_code) 용 얇은 래퍼.

범용 `huggingface` 챗 래퍼를 두 곳만 고쳐서 쓴다.
1) config.model_type 이 transformers 에 등록돼 있지 않으면 AutoModel 로 불러서 실패 → AutoModelForImageTextToText 로 강제.
2) 래퍼가 처리기에 넘기는 빈 videos/audios 인자를 SciEagleGraniteProcessor(→Siglip2 이미지 처리기)가 받지 못한다 → 빈 인자는 빼고 호출.
3) 처리기가 일반 dict 를 돌려주는데 래퍼는 .to(device) 를 부른다 → BatchFeature 로 감싼다.
4) 체크포인트의 tokenizer padding_side 는 right(학습용)인데, 디코더 모델의 배치 생성은 left 패딩이어야 한다.
   right 로 두면 batch_size>1 에서 짧은 프롬프트의 응답이 깨진다(chartqa 8문항 배치 4: 정답률 0.125, 배치 1: 정상) → 평가 래퍼에서만 left 로 강제한다.
"""

from transformers import AutoModelForImageTextToText, BatchFeature

import lmms_eval.models.chat.huggingface as _hf
from lmms_eval.api.registry import register_model


class _ProcessorShim:
    """processor 호출에서 비어 있는 videos/audios/images 인자를 정리하는 프록시. 그 외 속성은 그대로 위임."""

    def __init__(self, processor):
        object.__setattr__(self, "_p", processor)

    def __call__(self, *args, **kwargs):
        for k in ("videos", "audios", "images"):
            if k in kwargs and not kwargs[k]:
                kwargs.pop(k)
        out = self._p(*args, **kwargs)
        # SciEagleGraniteProcessor 는 일반 dict 를 돌려준다. 래퍼가 .to(device) 를 부르므로 BatchFeature 로 감싼다.
        return BatchFeature(out) if isinstance(out, dict) and not isinstance(out, BatchFeature) else out

    def __getattr__(self, name):
        return getattr(self._p, name)


@register_model("scieagle_granite")
class SciEagleGranite(_hf.Huggingface):
    def __init__(self, pretrained: str, trust_remote_code: bool = True, **kwargs):
        _hf.AutoModel = AutoModelForImageTextToText
        super().__init__(pretrained=pretrained, trust_remote_code=trust_remote_code, **kwargs)
        for tok in (getattr(self.processor, "tokenizer", None), getattr(self, "_tokenizer", None)):
            if tok is not None:
                tok.padding_side = "left"
        self.processor = _ProcessorShim(self.processor)
