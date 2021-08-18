import torch
from ctgan import CTGANSynthesizer, load_demo

if __name__ == "__main__":
    model = CTGANSynthesizer(epochs=1, cuda=torch.cuda.is_available())

    data = load_demo()
    discrete_columns = [
        'workclass',
        'education',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'native-country',
        'income'
    ]

    model.fit(data, discrete_columns)
    print(model._generator)
