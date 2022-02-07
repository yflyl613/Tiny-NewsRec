import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--news_query_vector_dim", type=int, default=200)
    parser.add_argument("--news_dim", type=int, default=256)
    parser.add_argument("--temp", type=float, default=0.05)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()
    return args
