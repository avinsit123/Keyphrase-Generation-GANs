import torch
from sequence_generator import SequenceGenerator
import logging
import config
from pykp.io import KeyphraseDataset
from torch.utils.data import DataLoader
import time
from utils.time_log import time_since
from evaluate import evaluate_beam_search
import pykp.io
import sys
import argparse
from utils.data_loader import load_data_and_vocab
from pykp.model import Seq2SeqModel
import os
import argparse
from torch import device

def init_pretrained_model(opt):
    model = Seq2SeqModel(opt)
    model.load_state_dict(torch.load(opt.model))
    """
    pretrained_state_dict = torch.load(opt.model)
    pretrained_state_dict_renamed = {}
    for k, v in pretrained_state_dict.items():
        if k.startswith("encoder.rnn."):
            k = k.replace("encoder.rnn.", "encoder.encoder.rnn.", 1)
        pretrained_state_dict_renamed[k] = v
    model.load_state_dict(pretrained_state_dict_renamed)
    """
    model.to(opt.device)
    model.eval()
    return model


def process_opt(opt):
    if opt.seed > 0:
        torch.manual_seed(opt.seed)

    if torch.cuda.is_available():
        if not opt.gpuid:
            opt.gpuid = 0
        opt.device = torch.device("cuda:%d" % opt.gpuid)
    else:
        opt.device = torch.device("cpu")
        opt.gpuid = -1
        print("CUDA is not available, fall back to CPU.")

    opt.exp = 'predict.' + opt.exp
    if opt.one2many:
        opt.exp += '.one2many'

    if opt.one2many_mode == 1:
        opt.exp += '.cat'

    if opt.copy_attention:
        opt.exp += '.copy'

    if opt.coverage_attn:
        opt.exp += '.coverage'

    if opt.review_attn:
        opt.exp += '.review'

    if opt.orthogonal_loss:
        opt.exp += '.orthogonal'

    if opt.use_target_encoder:
        opt.exp += '.target_encode'

    if hasattr(opt, 'bidirectional') and opt.bidirectional:
        opt.exp += '.bi-directional'
    else:
        opt.exp += '.uni-directional'

    if opt.n_best < 0:
        opt.n_best = opt.beam_size

    # fill time into the name
    if opt.exp_path.find('%s') > 0:
        opt.exp_path = opt.exp_path % (opt.exp, opt.timemark)
        opt.pred_path = opt.pred_path % (opt.exp, opt.timemark)

    if not os.path.exists(opt.exp_path):
        os.makedirs(opt.exp_path)
    if not os.path.exists(opt.pred_path):
        os.makedirs(opt.pred_path)

    if not opt.one2many and opt.one2many_mode > 0:
        raise ValueError("You cannot choose one2many mode without the -one2many options.")

    if opt.one2many and opt.one2many_mode == 0:
        raise ValueError("If you choose one2many, you must specify the one2many mode.")

    #if opt.greedy and not opt.one2many:
    #    raise ValueError("Greedy sampling can only be used in one2many mode.")

    if opt.one2many_mode not in [2, 3] and opt.max_eos_per_output_seq != 1:
        raise ValueError("You cannot specify the max_eos_per_output_seq unless your are using one2many_mode 2 or 3")

    return opt


def predict(test_data_loader, model, opt):
    if opt.delimiter_type == 0:
        delimiter_word = pykp.io.SEP_WORD
    else:
        delimiter_word = pykp.io.EOS_WORD
        
    generator = SequenceGenerator(model,
                                  bos_idx=opt.word2idx[pykp.io.BOS_WORD],
                                  eos_idx=opt.word2idx[pykp.io.EOS_WORD],
                                  pad_idx=opt.word2idx[pykp.io.PAD_WORD],
                                  peos_idx=opt.word2idx[pykp.io.PEOS_WORD],
                                  beam_size=1,
                                  max_sequence_length=opt.max_length,
                                  copy_attn=opt.copy_attention,
                                  coverage_attn=opt.coverage_attn,
                                  review_attn=opt.review_attn,
                                  cuda=opt.gpuid > -1
                                  )
    """
    if opt.one2many and opt.one2many_mode > 1:
        prediction_by_sampling(generator, test_data_loader, opt, delimiter_word)
    else:
        evaluate_beam_search(generator, test_data_loader, opt, delimiter_word)
    """
    if opt.sampling:
        raise ValueError("Not support yet!")
        #prediction_by_sampling(generator, test_data_loader, opt, delimiter_word)
    else:
        evaluate_beam_search(generator, test_data_loader, opt, delimiter_word)


def main(opt):
    try:
        start_time = time.time()
        load_data_time = time_since(start_time)
        test_data_loader, word2idx, idx2word, vocab = load_data_and_vocab(opt, load_train=False)
        model = init_pretrained_model(opt)
        logging.info('Time for loading the data and model: %.1f' % load_data_time)
        start_time = time.time()

        predict(test_data_loader, model, opt)

        total_testing_time = time_since(start_time)
        logging.info('Time for a complete testing: %.1f' % total_testing_time)
        print('Time for a complete testing: %.1f' % total_testing_time)
        sys.stdout.flush()

    except Exception as e:
        logging.exception("message")
    return

    pass

opt = argparse.Namespace(attn_mode='concat', include_attn_dist= 'True' ,baseline='self', batch_size=32, batch_workers=4, bidirectional=True, bridge='copy', checkpoint_interval=4000, copy_attention=True, copy_input_feeding=False, coverage_attn=False, coverage_loss=False, custom_data_filename_suffix=False, custom_vocab_filename_suffix=False, data='data/kp20k_separated/', data_filename_suffix='', dec_layers=1, decay_method='', decoder_size=300, decoder_type='rnn', delimiter_type=0, delimiter_word='<sep>', device=device(type='cuda', index=2), disable_early_stop_rl=False, dropout=0.1, dynamic_dict=True, early_stop_tolerance=4, enc_layers=1, encoder_size=150, encoder_type='rnn', epochs=20, exp='kp20k.rl.one2many.cat.copy.bi-directional', exp_path='exp/kp20k.rl.one2many.cat.copy.bi-directional.20190701-192604', final_perturb_std=0, fix_word_vecs_dec=False, fix_word_vecs_enc=False, goal_vector_mode=0, goal_vector_size=16, gpuid=1, init_perturb_std=0, input_feeding=False, lambda_coverage=1, lambda_orthogonal=0.03, lambda_target_encoder=0.03, learning_rate=0.001, learning_rate_decay=0.5, learning_rate_decay_rl=False, learning_rate_rl=5e-05, loss_normalization='tokens', manager_mode=1, match_type='exact', max_grad_norm=1, max_length=60, max_sample_length=6, max_unk_words=1000, mc_rollouts=False, model_path='model/kp20k.rl.one2many.cat.copy.bi-directional.20190701-192604', must_teacher_forcing=False, num_predictions=1, num_rollouts=3, one2many=True, one2many_mode=1, optim='adam', orthogonal_loss=False, param_init=0.1, perturb_baseline=False, perturb_decay_factor=0.0001, perturb_decay_mode=1, pre_word_vecs_dec=None, pre_word_vecs_enc=None, model='model/kp20k.ml.one2many.cat.copy.bi-directional.20190628-114655/kp20k.ml.one2many.cat.copy.bi-directional.epoch=2.batch=54573.total_batch=116000.model', regularization_factor=0.0, regularization_type=0, remove_src_eos=False, replace_unk=True, report_every=10, review_attn=False, reward_shaping=False, reward_type=7, save_model='model', scheduled_sampling=False, scheduled_sampling_batches=10000, seed=9527, separate_present_absent=True, share_embeddings=True, source_representation_queue_size=128, source_representation_sample_size=32, start_checkpoint_at=2, start_decay_at=8, start_epoch=1, target_encoder_size=64, teacher_forcing_ratio=0, timemark='20190701-192604', title_guided=False, topk='G', train_from='', train_ml=False, train_rl=True, truncated_decoder=0, use_target_encoder=False, vocab='data/kp20k_separated/', vocab_filename_suffix='', vocab_size=50002, warmup_steps=4000, word_vec_size=100, words_min_frequency=0 , beam_size = 5)


main(opt)
