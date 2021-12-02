import argparse
import utils
import logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        type=str,
                        default="train",
                        choices=['train', 'test'])
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default="../MIND/MINDlarge_train",
    )
    parser.add_argument(
        "--test_data_dir",
        type=str,
        default="../MIND/MINDlarge_test",
    )
    parser.add_argument("--filename_pat", type=str,
                        default="behaviors_np4_*.tsv")
    parser.add_argument("--model_dir", type=str, default='./model')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--npratio", type=int, default=4)
    parser.add_argument("--enable_gpu", type=utils.str2bool, default=True)
    parser.add_argument("--enable_hvd", type=utils.str2bool, default=True)
    parser.add_argument("--enable_shuffle", type=utils.str2bool, default=True)
    parser.add_argument("--shuffle_buffer_size", type=int, default=10000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--filter_num", type=int, default=3)
    parser.add_argument("--log_steps", type=int, default=100)

    # model training
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0001)

    parser.add_argument("--num_words_title", type=int, default=20)
    parser.add_argument("--num_words_abstract", type=int, default=50)
    parser.add_argument("--num_words_body", type=int, default=100)
    parser.add_argument(
        "--user_log_length",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--word_embedding_dim",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--glove_embedding_path",
        type=str,
        default='/home/v-yangyu1/glove.840B.300d.txt',
    )
    parser.add_argument("--freeze_embedding",
                        type=utils.str2bool,
                        default=False)
    parser.add_argument(
        "--news_dim",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--news_query_vector_dim",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--user_query_vector_dim",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--num_attention_heads",
        type=int,
        default=20,
    )
    parser.add_argument("--user_log_mask", type=utils.str2bool, default=True)
    parser.add_argument("--drop_rate", type=float, default=0.2)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--max_steps_per_epoch", type=int, default=1000000)

    parser.add_argument(
        "--load_ckpt_name",
        type=str,
        default=None,
        help="choose which ckpt to load and test"
    )

    # bert
    parser.add_argument("--apply_bert", type=utils.str2bool, default=False)
    parser.add_argument("--model_type", default="bert", type=str)
    parser.add_argument("--do_lower_case", type=utils.str2bool, default=True)
    parser.add_argument(
        "--model_name", default="../bert-base-uncased/pytorch_model.bin", type=str)
    parser.add_argument(
        "--config_name", default="../bert-base-uncased/config.json", type=str)
    parser.add_argument("--tokenizer_name",
                        default="../bert-base-uncased/vocab.txt", type=str)

    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument(
        "--bert_trainable_layer",
        type=int, nargs='+',
        default=[],
        choices=list(range(12)))

    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--pooling", type=str, default='att')
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--use_pretrain_model",
                        type=utils.str2bool, default=False)
    parser.add_argument("--pretrain_model_path", type=str, default=None)
    parser.add_argument("--pretrain_lr", type=float, default=0.00001)
    parser.add_argument("--num_teacher_layers", type=int, default=12)
    parser.add_argument("--num_student_layers", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--coef", type=float, default=1.0)
    parser.add_argument("--tensorboard", type=str, default=None)
    parser.add_argument("--teacher_ckpt", type=str, default=None)

    args = parser.parse_args()

    logging.info(args)
    return args


if __name__ == "__main__":
    args = parse_args()
