import torch
from torch import nn

from sparse2unseen.ssl.ema import make_teacher, update_ema


def test_ema_update():
    student = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        student.weight.fill_(2.0)
    teacher = make_teacher(student)
    with torch.no_grad():
        student.weight.fill_(4.0)
    update_ema(student, teacher, decay=0.5)
    assert torch.allclose(teacher.weight, torch.full_like(teacher.weight, 3.0))
