test_dict = {
    "task": 123,
    "test": [1, 2, 3]
}

if __name__ == '__main__':
    obs = {
        **test_dict
    }
    print(obs["test"])
