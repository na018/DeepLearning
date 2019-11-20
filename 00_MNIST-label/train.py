import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a simple neural net to recognize number images from the MNIST dataset and appy the correct labeld')
    parser.add_argument('--batch_amount', default='200',
                        help='Amount of batches the net trains on')
    parser.add_argument('--batch_size', default='100',
                        help='Number of training samples inside one batch')
    parser.add_argument('--lr', default='0.01', help='Learning Rate')
