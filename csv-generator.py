import os
import string

import pandas as pd
import pytesseract
from PIL import Image, ImageFilter, ImageOps

relevant_words = {
    "login": ["username", "user", "email", "mail", "address", "password", "login", "log", "in", "signin", "sign", "forgot",  "logging", "signing", "phone", "number", "keep", "signed", "remember"],
    "register": ["name", "email", "username", "mail", "address", "password", "surname", "phone", "number", "first", "last", "submit", "register", "registration", "signup", "sign", "up", "create", "join", "confirm", "terms", "started", "get", "already", "account"],
    "payment": ["payment", "pay", "card", "credit", "number", "debit", "expiration", "date", "cvc", "mm", "yy", "cvv", "pay", "checkout", "order", "purchase", "buy", "voucher", "total", "subtotal", "month", "year", "holder"]
}

def get_words_from_image(image_path, type):

    image = Image.open(image_path)
    # gri
    gray_image = image.convert('L')

    unsharp_mask = gray_image.filter(ImageFilter.UnsharpMask(radius=8, percent=300, threshold=0))
    extracted_text = pytesseract.image_to_string(unsharp_mask)

    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))  # map punctuation to space
    extracted_text = extracted_text.translate(translator)

    extracted_text = extracted_text.replace('\n', ' ')
    extracted_text = extracted_text.lower()
    words = extracted_text.split(" ")

    filtered_words = []
    for word in words:
        if word in relevant_words[type]:
            filtered_words.append(word)

    filtered_text = ' '.join(filtered_words)

    return filtered_text


def load_data():
    login_directory = '/home/luiza/Documents/Licenta/git/licenta/Licenta/dataset bun/LoginScreenshots'
    register_directory = '/home/luiza/Documents/Licenta/git/licenta/Licenta/dataset bun/RegisterScreenshots'
    payment_directory = '/home/luiza/Documents/Licenta/git/licenta/Licenta/dataset bun/PaymentScreenshots'

    login_images = []
    login_images_text = []
    login_classes = []
    print("**********LOGIN*********")
    for filename in os.listdir(login_directory):
        print(filename)
        image_path = login_directory + "/" + filename
        image_text = get_words_from_image(image_path, "login")
        login_images.append(filename)
        login_classes.append("login")
        login_images_text.append(image_text)


    register_images = []
    register_images_text = []
    register_classes = []
    print("**********REGISTER*********")
    for filename in os.listdir(register_directory):
        print(filename)
        image_path = register_directory + "/" + filename
        image_text = get_words_from_image(image_path, "register")
        register_images.append(filename)
        register_classes.append("register")
        register_images_text.append(image_text)


    payment_images = []
    payment_images_text = []
    payment_classes = []
    print("**********PAYMENT*********")
    for filename in os.listdir(payment_directory):
        print(filename)
        image_path = payment_directory + "/" + filename
        image_text = get_words_from_image(image_path, "payment")
        login_images.append(filename)
        payment_classes.append("payment")
        payment_images_text.append(image_text)

    print(len(login_images), len(login_classes))
    print(len(register_images), len(register_classes))
    print(len(payment_images), len(payment_classes))


    train_X_image = []
    train_X_text = []
    train_Y = []

    test_X_image = []
    test_X_text = []
    test_Y = []

    train_X_image.extend(login_images[:int(0.9 * len(login_images))])
    train_X_text.extend(login_images_text[:int(0.9 * len(login_images_text))])
    train_Y.extend(login_classes[:int(0.9 * len(login_classes))])

    train_X_image.extend(register_images[:int(0.9 * len(register_images))])
    train_X_text.extend(register_images_text[:int(0.9 * len(register_images_text))])
    train_Y.extend(register_classes[:int(0.9 * len(register_classes))])

    train_X_image.extend(payment_images[:int(0.9 * len(payment_images))])
    train_X_text.extend(payment_images_text[:int(0.9 * len(payment_images_text))])
    train_Y.extend(payment_classes[:int(0.9 * len(payment_classes))])



    test_X_image.extend(login_images[int(0.9 * len(login_images)):])
    test_X_text.extend(login_images_text[int(0.9 * len(login_images_text)):])
    test_Y.extend(login_classes[int(0.9 * len(login_classes)):])

    test_X_image.extend(register_images[int(0.9 * len(register_images)):])
    test_X_text.extend(register_images_text[int(0.9 * len(register_images_text)):])
    test_Y.extend(register_classes[int(0.9 * len(register_classes)):])

    test_X_image.extend(payment_images[int(0.9 * len(payment_images)):])
    test_X_text.extend(payment_images_text[int(0.9 * len(payment_images_text)):])
    test_Y.extend(payment_classes[int(0.9 * len(payment_classes)):])

    # csv
    # dictionary of lists
    dict = {'text': train_X_text, 'img_path': train_X_image, 'label': train_Y}
    df = pd.DataFrame(dict)
    # saving the dataframe
    df.to_csv('train.csv')

    dict = {'text': test_X_text, 'img_path': test_X_image, 'label': test_Y}
    df = pd.DataFrame(dict)
    # saving the dataframe
    df.to_csv('test.csv')


load_data()