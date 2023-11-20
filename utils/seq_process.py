import torch

# seq's shape [bs 263 1 T]
def get_canavas(seq_0, args_0, seq_1, args_1, inter_frames):
    # seq_canavas is the background
    bs, njoints, nfeats, max_frames_0 = seq_0.shape
    _, _, _, max_frames_1 = seq_1.shape
    max_frames = max_frames_0 + max_frames_1 - inter_frames
    # len = args_0['length'] + args_1['length'] - inter_frames
    seq_canavas = torch.zeros((bs, njoints, nfeats, max_frames), device=seq_0.device)
    # count record the weight
    count = torch.zeros((bs, max_frames), device=seq_0.device)

    # TODO convert to tensor operation
    for idx in range(bs):
        len_0 = args_0['length'][idx]
        len_1 = args_1['length'][idx]
        len = len_0 + len_1 - inter_frames
        seq_canavas[idx,:,:,:len_0] += seq_0[idx,:,:,:len_0]
        seq_canavas[idx,:,:,len_0-inter_frames:len] += seq_1[idx,:,:,:len_1]
        count[idx,:len_0] += 1 
        count[idx, len_0-inter_frames:len] += 1

    count = count.unsqueeze(1).unsqueeze(1)
    seq_comp = seq_canavas / (count + 1e-5)

    return seq_comp

def sync_fn(seq_0, args_0, seq_1, args_1, inter_frames):
    
    seq_comp = get_canavas(seq_0, args_0, seq_1, args_1, inter_frames)

    # ret_seq_0 = torch.zeros_like(seq_0)
    # ret_seq_1 = torch.zeros_like(seq_1)

    # for idx in range(bs):
    #     len_0 = args_0['length'][idx]
    #     len_1 = args_1['length'][idx]
    #     len = len_0 + len_1 - inter_frames
    #     ret_seq_0[idx,:,:,:len_0] = seq_comp[idx,:,:,:len_0]
    #     ret_seq_1[idx,:,:,:len_1] = seq_comp[idx,:,:,len_0-inter_frames:len]

    return extract_fn(seq_comp, args_0, args_1, inter_frames)

def extract_fn(seq_comp, args_0, args_1, inter_frames):
    bs, njoints, nfeats, _ = seq_comp.shape

    ret_seq_0 = torch.zeros((bs, njoints, nfeats, max(args_0['length'])), device=seq_comp.device)
    ret_seq_1 = torch.zeros((bs, njoints, nfeats, max(args_1['length'])), device=seq_comp.device)

    for idx in range(bs):
        len_0 = args_0['length'][idx]
        len_1 = args_1['length'][idx]
        len = len_0 + len_1 - inter_frames
        ret_seq_0[idx,:,:,:len_0] = seq_comp[idx,:,:,:len_0]
        ret_seq_1[idx,:,:,:len_1] = seq_comp[idx,:,:,len_0-inter_frames:len]

    return ret_seq_0, ret_seq_1
    