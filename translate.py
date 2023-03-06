"""
This file allow you to test the translator.
The configuration file of the model that is going to be used to translate must be specified.
First create that configuration file in model_configs directory using the "example.py" file template.
Besides write the input text in a file.
Then execute this file while specifying the model configuration file, the input text file path and the output text file in command line as follow:
python translate.py model_configs/<config_file> --input_file=<input_text_path> --output_file=<output_text_file>

<config_file> is a placeholder of the used model's configuration file.
<input_text_path> and <output_text_file> are placeholders for the actual values.

You can also use the default model. In that case just specify input text and output text files as follow:
python translate.py --input_file=<input_text_path> --output_file=<output_text_file>
"""

from model import Translator, TranslatorConfig
import os

ROOT = os.getenv("ROOT")
if ROOT is None:
    raise Exception("Error: environment variable ROOT must be set. Please execute setup.py to solve this problem.")

#----------------------------------
# Initializing input_file and output_file in the globlas() with dummy values
input_file = None
output_file = None
config = None
# Taking the values specified in command line
exec(open(os.path.join(ROOT, "arg_parser.py"), "r").read())
#----------------------------------

# Defining the model
if config is None:
    print("Using the default model to translate")
    config = TranslatorConfig()
else:
    config = TranslatorConfig(**config)
    print(f"Using the model in {config.model_path} to translate")

translator = Translator(config)

# Reading the input text
input_text = open(input_file, "r").read()

# Translation
print(f"Translating text in {input_file}, result in {output_file} ...")
output = translator([input_text])

with open(output_file, 'w') as f:
    f.write(output[0])

print("Translation completed.")