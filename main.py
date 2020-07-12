import DataPreprocessing
import LayerOptimizer
from sklearn.model_selection import train_test_split

file = 'proteindata.csv'


def main():
    pseqs, scaled_values = DataPreprocessing.run(file)
    opt_model, split = LayerOptimizer.run(pseqs, scaled_values)
    train_x, test_x, train_y, test_y = split
    opt_model.evaluate(test_x, test_y)
    print("Done")


if __name__ == "__main__":
    main()


