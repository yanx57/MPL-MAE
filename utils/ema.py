import copy
import torch

class EMA:
    def __init__(self, model, decay):
        """
        model: 你的 base_model
        decay: 衰减率，建议 0.999 或 0.9999
        """
        self.model = model
        # 深拷贝一个影子模型，用于存储平滑后的参数
        self.shadow = copy.deepcopy(self.model)
        self.decay = decay
        
        # 影子模型不需要梯度
        for param in self.shadow.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self):
        """每个 iteration 结束后续调用此函数"""
        # 参数更新公式: shadow = decay * shadow + (1 - decay) * online
        for shadow_param, online_param in zip(self.shadow.parameters(), self.model.parameters()):
            shadow_param.data.mul_(self.decay).add_(online_param.data, alpha=1 - self.decay)
            
        # 如果模型中有 Buffer（如 BatchNorm 的 running_mean），也需要同步
        for shadow_buffer, online_buffer in zip(self.shadow.buffers(), self.model.buffers()):
            shadow_buffer.copy_(online_buffer)

    def apply_shadow(self):
        """验证时：将影子模型的参数覆盖到当前模型（暂存原始参数以便恢复）"""
        self.backup = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(self.shadow.state_dict())

    def restore(self):
        """验证结束：恢复原始参数，继续训练"""
        self.model.load_state_dict(self.backup)