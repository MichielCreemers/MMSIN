import argparse
import time

from utils import projections



def assess_quality(config):
    start = time.time()
    
    end = time.time()

if __name__ == '__main__':

    parser  = argparse.ArgumentParser()

    parser.add_argument('--pcname', type=str, help='path to the point cloud whose quality we want to assess')
    parser.add_argument('--model', type=str, help='path to the trained model we want to use to assess the point cloud')

    config = parser.parse_args()

    asses_quality(config)