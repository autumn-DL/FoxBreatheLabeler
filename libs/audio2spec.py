import numpy as np
import torch
import torch.nn as nn


def get_music_chunk(
        y,
        *,
        frame_length=2048,
        hop_length=512,
        pad_mode="constant",
):
    '''

    :param y: T
    :param frame_length: int
    :param hop_length: int
    :param pad_mode:
    :return: T
    '''
    # padding = (int(frame_length // 2), int(frame_length // 2))
    padding = (int((frame_length - hop_length) // 2),
               int((frame_length - hop_length + 1) // 2))

    y = torch.nn.functional.pad(y, padding, pad_mode)
    y_f = y.unfold(0, frame_length, hop_length)

    return y_f


class get_music_chunk_spec(nn.Module):
    def __init__(self, frame_length=2048,
                 hop_length=512,
                 pad_mode="constant", ):
        super().__init__()
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.pad_mode = pad_mode
        self.padding = (int((self.frame_length - self.hop_length) // 2),
                        int((self.frame_length - self.hop_length + 1) // 2))
        # self.get_music_chunk=get_music_chunk

    def forward(self, x):
        y = torch.nn.functional.pad(x, self.padding, self.pad_mode)
        y_f = y.unfold(0, self.frame_length, self.hop_length)

        return y_f


class batch_get_music_unchunk(nn.Module):
    def __init__(self, frame_length=2048,
                 hop_length=512,
                 pad_mode="constant", ):
        super().__init__()
        F_num = int(frame_length - hop_length)
        self.MFC = False
        if frame_length - 2 * F_num < 0:
            self.MFC = True
        if not self.MFC:
            W1, W2, W3 = G_add_weige(frame_length=frame_length, hop_length=hop_length)
            self.register_buffer('W1', W1, persistent=False)
            self.register_buffer('W2', W2, persistent=False)
            self.register_buffer('W3', W3, persistent=False)
        self.padding = (int((frame_length - hop_length) // 2),
                        int((frame_length - hop_length + 1) // 2))
        self.pad_mode = pad_mode
        self.frame_length = frame_length
        self.hop_length = hop_length

    def get_music_unchunk_fast(self, y):
        '''

        :param y:F T
        :param frame_length: int
        :param hop_length: int
        :param pad_mode:
        :return: T
        '''
        # padding = (int(frame_length // 2), int(frame_length // 2))

        FV, _ = y.shape

        yx = y
        yp = torch.zeros_like(yx)
        if not self.MFC:
            yp[0] = yx[0] * self.W1

            yp[-1] = yx[-1] * self.W3

            yp[1:-1, :] = yx[1:-1, :] * self.W2.unsqueeze(0)
        else:
            yp = yx
        # y_f=y.unfold(0, frame_length, hop_length)
        outsc = torch.zeros(1, len(yp) * self.hop_length + self.padding[0] + self.padding[1], dtype=y.dtype,
                            device=y.device)
        # st = 0
        # for i in yp:
        #     outsc[(st) * self.hop_length:(st) * self.hop_length + self.frame_length] += i #i-T outsc-T1
        #     st += 1

        # indices = torch.arange(0, FV, device=yp.device)
        # indices = torch.arange(0, self.frame_length, device=yp.device)
        # sy = yp.index_select(dim=1, index=indices).squeeze(1)
        # sy = yp

        hop_offsets = torch.arange(0, self.hop_length * FV, self.hop_length, device=yp.device)
        FB = torch.arange(0, self.frame_length, device=yp.device).repeat(FV, 1) + hop_offsets[:, None]
        FB = FB.view(1, -1)
        yp = yp.view(1, -1)

        # frame_offsets = hop_offsets + self.frame_length
        # outsc[ hop_offsets:frame_offsets,indices] += sy
        outsc.scatter_add_(1, FB, yp)
        if self.MFC:
            spx = torch.zeros_like(outsc)
            spx.scatter_add_(1, FB, torch.ones_like(yp))
            spx = torch.clamp(spx, 1, None).float()
            outsc /= spx
        y = torch.nn.functional.pad(outsc[0], (-self.padding[0], -self.padding[1]), self.pad_mode)

        return y

    def batchs_get_music_unchunk_fast(self, y):
        '''

        :param y: B F T
        :param frame_length: int
        :param hop_length: int
        :param pad_mode:
        :return: B T
        '''
        # padding = (int(frame_length // 2), int(frame_length // 2))
        BV, FV, _ = y.shape
        yx = y
        yp = torch.zeros_like(yx)
        if not self.MFC:
            yp[:, 0] = yx[:, 0] * self.W1

            yp[:, -1] = yx[:, -1] * self.W3

            yp[:, 1:-1, :] = yx[:, 1:-1, :] * self.W2.unsqueeze(0)
        else:
            yp = yx
        # y_f=y.unfold(0, frame_length, hop_length)
        outsc = torch.zeros(BV, FV * self.hop_length + self.padding[0] + self.padding[1], dtype=y.dtype,
                            device=y.device)

        hop_offsets = torch.arange(0, self.hop_length * FV, self.hop_length, device=yp.device)
        FB = torch.arange(0, self.frame_length, device=yp.device).repeat(FV, 1) + hop_offsets[:, None]
        FB = FB.view(1, -1).repeat(BV, 1)
        yp = yp.view(BV, -1)

        # frame_offsets = hop_offsets + self.frame_length
        # outsc[ hop_offsets:frame_offsets,indices] += sy
        outsc.scatter_add_(1, FB, yp)

        if self.MFC:
            spx = torch.zeros(BV, FV * self.hop_length + self.padding[0] + self.padding[1], dtype=y.dtype,
                              device=y.device)

            spx.scatter_add_(1, FB, torch.ones_like(yp))
            spx = torch.clamp(spx, 1, None).float()
            outsc /= spx

        y = torch.nn.functional.pad(outsc, (-self.padding[0], -self.padding[1]), self.pad_mode)

        return y

    def forward(self, x, mask=None):
        if mask is None:
            if len(x.size()) == 3:
                y = self.batchs_get_music_unchunk_fast(x)
            else:
                # y = self.get_music_unchunk(x)
                y = self.get_music_unchunk_fast(x)
        else:
            ML = mask.sum(1)
            # y = self.get_music_unchunk(x) * mask
            spx = []
            for i, im in zip(x, ML):
                spx.append(
                    torch.nn.functional.pad(self.get_music_unchunk_fast(i)[:int(im)], (0, ML * self.hop_length), ))
            y = torch.stack(spx)

        return y


def G_add_weige(frame_length=2048, hop_length=512, ):
    F_num = int(frame_length - hop_length)

    W1 = (1 / (F_num - 1)) * torch.arange(F_num)
    W2 = (1 / (F_num - 1)) * (torch.arange(F_num - 1, -1, step=-1))

    if frame_length - 2 * F_num == 0:
        return torch.cat([torch.ones(F_num), W1]), torch.cat([W2, W1]), torch.cat([W2, torch.ones(F_num)])
    else:
        return torch.cat([torch.ones(frame_length - F_num), W1]), torch.cat(
            [W2, torch.ones(frame_length - 2 * F_num), W1]), torch.cat([W2, torch.ones(frame_length - F_num)])


if __name__ == '__main__':
    pass

    sssx = get_music_chunk(torch.tensor([i for i in range(111)]), hop_length=10, frame_length=15).float()
    ssp = G_add_weige(hop_length=10, frame_length=15)
    bc = batch_get_music_unchunk(hop_length=10, frame_length=15)
    nc = bc(torch.cat([sssx.unsqueeze(0), sssx.unsqueeze(0)], dim=0))
    # nc = bc(sssx)
    pass
