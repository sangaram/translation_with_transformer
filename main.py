from model import TranslatorConfig, Translator


if __name__ == '__main__':
    config = TranslatorConfig()
    translator = Translator(config)

    input = ["This is a test"]
    output = translator(input)
    print(f"Input: {input},\tOuput: {output}")
    
