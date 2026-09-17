from __future__ import annotations

import copy

import torch
from torch import nn


@torch.no_grad()
def update_ema(student: nn.Module, teacher: nn.Module, decay: float) -> None:
    student_params = dict(student.named_parameters())
    for name, teacher_param in teacher.named_parameters():
        teacher_param.mul_(decay).add_(student_params[name], alpha=1.0 - decay)
    student_buffers = dict(student.named_buffers())
    for name, teacher_buffer in teacher.named_buffers():
        teacher_buffer.copy_(student_buffers[name])


def make_teacher(student: nn.Module) -> nn.Module:
    teacher = copy.deepcopy(student)
    teacher.requires_grad_(False)
    teacher.eval()
    return teacher
