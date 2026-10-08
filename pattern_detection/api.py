"""
功 能：攻击手法检测api入口
"""

from classifier_config import ClassifierConfig
from identify_positive_by_model import PatternIdentification


class AttackPatternDetectAPI:
    """
    攻击手法检测类
    """
    def __init__(self, logger):
        self.logger = logger
        self.pattern_identification = self.init_detection()

    def init_detection(self):
        """
        检测相关模型初始化
        """
        try:
            init_obj = PatternIdentification(self.logger)
            init_obj.init_main()
            self.logger.info("[AI Guardrail] agent pattern detection: Init successfully.")
            return init_obj
        except Exception as e:
            self.logger.error("[AI Guardrail] agent pattern detection: Init failed.")
            raise RuntimeError("Init Error") from e

    def detect(self, content):
        """
        检测实现
        :param content: 输入待检文本
        :return 返回检测结果
        """
        try:
            detection_flag = self.pattern_identification.filter_by_extract_features(content)
            if detection_flag == ClassifierConfig.POSITIVE_FLAG:
                # 接口定义score取值0.0代表无风险
                return {"score": 0.0, "evidence": ""}
            elif detection_flag == ClassifierConfig.NEGATIVE_FLAG:
                # 接口定义score取值为1.0代表有风险
                return {"score": 1.0, "evidence": ""}
            else:
                # 接口定义score取值0.5代表存疑
                return {"score": 0.5, "evidence": ""}
        except Exception as e:
            self.logger.error("[AI Guardrail] agent pattern detection: Detect Exception.")
            raise RuntimeError("Detect Exception") from e
