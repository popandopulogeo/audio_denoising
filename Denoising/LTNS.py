from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, BatchNormalization, ReLU, Input, LSTM, Concatenate, Conv2DTranspose

def SID_component(sid_input):

    conv1 = Conv2D(filters=48, kernel_size=(1,7), dilation_rate=(1,1))(sid_input)
    conv1 = BatchNormalization()(conv1)
    conv1 = ReLU()(conv1)

    conv2 = Conv2D(filters=48, kernel_size=(7,1), dilation_rate=(1,1))(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = ReLU()(conv2)

    conv3 = Conv2D(filters=48, kernel_size=(5,5), dilation_rate=(1,1))(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = ReLU()(conv3)

    conv4 = Conv2D(filters=48, kernel_size=(5,5), dilation_rate=(2,1))(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = ReLU()(conv4)

    conv5 = Conv2D(filters=48, kernel_size=(5,5), dilation_rate=(4,1))(conv4)
    conv5 = BatchNormalization()(conv5)
    conv5 = ReLU()(conv5)

    conv6 = Conv2D(filters=48, kernel_size=(5,5), dilation_rate=(8,1))(conv5)
    conv6 = BatchNormalization()(conv6)
    conv6 = ReLU()(conv6)

    conv7 = Conv2D(filters=48, kernel_size=(5,5), dilation_rate=(16,1))(conv6)
    conv7 = BatchNormalization()(conv7)
    conv7 = ReLU()(conv7)

    conv8 = Conv2D(filters=48, kernel_size=(5,5), dilation_rate=(32,1))(conv7)
    conv8 = BatchNormalization()(conv8)
    conv8 = ReLU()(conv8)

    conv9 = Conv2D(filters=48, kernel_size=(5,5), dilation_rate=(1,1))(conv8)
    conv9 = BatchNormalization()(conv9)
    conv9 = ReLU()(conv9)

    conv10 = Conv2D(filters=48, kernel_size=(5,5), dilation_rate=(2,2))(conv9)
    conv10 = BatchNormalization()(conv10)
    conv10 = ReLU()(conv10)

    conv11 = Conv2D(filters=48, kernel_size=(5,5), dilation_rate=(4,4))(conv10)
    conv11 = BatchNormalization()(conv11)
    conv11 = ReLU()(conv11)

    conv12 = Conv2D(filters=48, kernel_size=(1,1), dilation_rate=(1,1))(conv11)
    conv12 = BatchNormalization()(conv12)
    conv12 = ReLU()(conv12)

    lstm = LSTM(units=100)(conv2)

    fc = Dense(units=100)(lstm)
    fc = ReLU()(fc)

    sid_output = Dense(units=1)(fc)

    return sid_output

def NE_component(noisy_signal, noise_profile):

    #First encoder
    conv1_1 = Conv2D(filters=64, kernel_size=5, dilation_rate=1, stride=1)(noisy_signal)
    conv1_1 = BatchNormalization()(conv1_1)
    conv1_1 = ReLU()(conv1_1)

    conv2_1 = Conv2D(filters=128, kernel_size=5, dilation_rate=1, stride=2)(conv1_1)
    conv2_1 = BatchNormalization()(conv2_1)
    conv2_1 = ReLU()(conv2_1)

    conv3_1 = Conv2D(filters=128, kernel_size=5, dilation_rate=1, stride=1)(conv2_1)
    conv3_1 = BatchNormalization()(conv3_1)
    conv3_1 = ReLU()(conv3_1)

    #Second encoder
    conv1_2 = Conv2D(filters=64, kernel_size=5, dilation_rate=1, stride=1)(noise_profile)
    conv1_2 = BatchNormalization()(conv1_2)
    conv1_2 = ReLU()(conv1_2)

    conv2_2 = Conv2D(filters=128, kernel_size=5, dilation_rate=1, stride=2)(conv1_2)
    conv2_2 = BatchNormalization()(conv2_2)
    conv2_2 = ReLU()(conv2_2)

    conv3_2 = Conv2D(filters=128, kernel_size=5, dilation_rate=1, stride=1)(conv2_2)
    conv3_2 = BatchNormalization()(conv3_2)
    conv3_2 = ReLU()(conv3_2)

    encoders_output = Concatenate([conv3_1, conv3_2])

    #Decoder
    uconv1 = Conv2D(filters=256, kernel_size=3, dilation_rate=1, stride=2)(encoders_output)
    uconv1 = BatchNormalization()(uconv1)
    uconv1 = ReLU()(uconv1)

    uconv2 = Conv2D(filters=256, kernel_size=3, dilation_rate=1, stride=1)(uconv1)
    uconv2 = BatchNormalization()(uconv2)
    uconv2 = ReLU()(uconv2)

    uconv3 = Conv2D(filters=256, kernel_size=3, dilation_rate=2, stride=1)(uconv2)
    uconv3 = BatchNormalization()(uconv3)
    uconv3 = ReLU()(uconv3)
    
    uconv4 = Conv2D(filters=256, kernel_size=3, dilation_rate=4, stride=1)(uconv3)
    uconv4 = BatchNormalization()(uconv4)
    uconv4 = ReLU()(uconv4)

    uconv5 = Conv2D(filters=256, kernel_size=3, dilation_rate=8, stride=1)(uconv4)
    uconv5 = BatchNormalization()(uconv5)
    uconv5 = ReLU()(uconv5)

    uconv6 = Conv2D(filters=256, kernel_size=3, dilation_rate=16, stride=1)(uconv5)
    uconv6 = BatchNormalization()(uconv6)
    uconv6 = ReLU()(uconv6)

    uconv7 = Conv2D(filters=256, kernel_size=3, dilation_rate=1, stride=1)(uconv6)
    uconv7 = BatchNormalization()(uconv7)
    uconv7 = ReLU()(uconv7)

    uconv8 = Conv2D(filters=256, kernel_size=3, dilation_rate=1, stride=1)(uconv7)
    uconv8 = BatchNormalization()(uconv8)
    uconv8 = ReLU()(uconv8)

    uconv9 = Conv2DTranspose(filters=128, kernel_size=3, dilation_rate=1, stride=2)(uconv8)
    uconv9 = Concatenate([uconv1, uconv9])
    uconv9 = BatchNormalization()(uconv9)
    uconv9 = ReLU()(uconv9)

    uconv10 = Conv2D(filters=128, kernel_size=3, dilation_rate=1, stride=1)(uconv9)
    uconv10 = BatchNormalization()(uconv10)
    uconv10 = ReLU()(uconv10)

    uconv11 = Conv2DTranspose(filters=64, kernel_size=3, dilation_rate=1, stride=2)(uconv10)
    uconv11 = Concatenate([conv2_1, conv2_2, uconv11])
    uconv11 = BatchNormalization()(uconv11)
    uconv11 = ReLU()(uconv11)

    uconv12 = Conv2D(filters=64, kernel_size=3, dilation_rate=1, stride=1)(uconv11)
    uconv12 = BatchNormalization()(uconv12)
    uconv12 = ReLU()(uconv12)

    ne_output = Conv2D(filters=2, kernel_size=3, dilation_rate=1, stride=1)(uconv12)

    return ne_output

def NR_component(audio_spectr, noise_spectr):

    #First encoder
    conv1_1 = Conv2D(filters=96, kernel_size=(1,7), dilation_rate=(1,1), stride=1)(audio_spectr)
    conv1_1 = BatchNormalization()(conv1_1)
    conv1_1 = ReLU()(conv1_1)

    conv2_1 = Conv2D(filters=96, kerel_size=(7,1), dilation_rate=(1,1), stride=1)(conv1_1)
    conv2_1 = BatchNormalization()(conv2_1)
    conv2_1 = ReLU(conv2_1)

    conv3_1 = Conv2D(filters=96, kerel_size=(5,5), dilation_rate=(1,1), stride=1)(conv2_1)
    conv3_1 = BatchNormalization()(conv3_1)
    conv3_1 = ReLU(conv3_1)

    conv4_1 = Conv2D(filters=96, kerel_size=(5,5), dilation_rate=(2,1), stride=1)(conv3_1)
    conv4_1 = BatchNormalization()(conv4_1)
    conv4_1 = ReLU(conv4_1)

    conv5_1 = Conv2D(filters=96, kerel_size=(5,5), dilation_rate=(4,1), stride=1)(conv4_1)
    conv5_1 = BatchNormalization()(conv5_1)
    conv5_1 = ReLU(conv5_1)

    conv6_1 = Conv2D(filters=96, kerel_size=(5,5), dilation_rate=(8,1), stride=1)(conv5_1)
    conv6_1 = BatchNormalization()(conv6_1)
    conv6_1 = ReLU(conv6_1)

    conv7_1 = Conv2D(filters=96, kerel_size=(5,5), dilation_rate=(16,1), stride=1)(conv6_1)
    conv7_1 = BatchNormalization()(conv7_1)
    conv7_1 = ReLU(conv7_1)

    conv8_1 = Conv2D(filters=96, kerel_size=(5,5), dilation_rate=(32,1), stride=1)(conv7_1)
    conv8_1 = BatchNormalization()(conv8_1)
    conv8_1 = ReLU(conv8_1)

    conv9_1 = Conv2D(filters=96, kerel_size=(5,5), dilation_rate=(1,1), stride=1)(conv8_1)
    conv9_1 = BatchNormalization()(conv9_1)
    conv9_1 = ReLU(conv9_1)

    conv10_1 = Conv2D(filters=96, kerel_size=(5,5), dilation_rate=(2,2), stride=1)(conv9_1)
    conv10_1 = BatchNormalization()(conv10_1)
    conv10_1 = ReLU(conv10_1)

    conv11_1 = Conv2D(filters=96, kerel_size=(5,5), dilation_rate=(4,4), stride=1)(conv10_1)
    conv11_1 = BatchNormalization()(conv11_1)
    conv11_1 = ReLU(conv11_1)

    conv12_1 = Conv2D(filters=96, kerel_size=(5,5), dilation_rate=(8,8), stride=1)(conv11_1)
    conv12_1 = BatchNormalization()(conv12_1)
    conv12_1 = ReLU(conv12_1)

    conv13_1 = Conv2D(filters=96, kerel_size=(5,5), dilation_rate=(16,16), stride=1)(conv12_1)
    conv13_1 = BatchNormalization()(conv13_1)
    conv13_1 = ReLU(conv13_1)

    conv14_1 = Conv2D(filters=96, kerel_size=(5,5), dilation_rate=(32,32), stride=1)(conv13_1)
    conv14_1 = BatchNormalization()(conv14_1)
    conv14_1 = ReLU(conv14_1)

    conv15_1 = Conv2D(filters=8, kerel_size=(1,1), dilation_rate=(1,1), stride=1)(conv14_1)
    conv15_1 = BatchNormalization()(conv15_1)
    conv15_1 = ReLU(conv15_1)

    #Second encoder
    conv1_2 = Conv2D(filters=48, kernel_size=(1,7), dilation_rate=(1,1), stride=1)(noise_spectr)
    conv1_2 = BatchNormalization()(conv1_2)
    conv1_2 = ReLU()(conv1_2)

    conv2_2 = Conv2D(filters=48, kerel_size=(7,1), dilation_rate=(1,1), stride=1)(conv1_2)
    conv2_2 = BatchNormalization()(conv2_2)
    conv2_2 = ReLU(conv2_2)

    conv3_2 = Conv2D(filters=48, kerel_size=(5,5), dilation_rate=(1,1), stride=1)(conv2_2)
    conv3_2 = BatchNormalization()(conv3_2)
    conv3_2 = ReLU(conv3_2)

    conv4_2 = Conv2D(filters=48, kerel_size=(5,5), dilation_rate=(2,1), stride=1)(conv3_2)
    conv4_2 = BatchNormalization()(conv4_2)
    conv4_2 = ReLU(conv4_2)

    conv5_2 = Conv2D(filters=48, kerel_size=(5,5), dilation_rate=(4,1), stride=1)(conv4_2)
    conv5_2 = BatchNormalization()(conv5_2)
    conv5_2 = ReLU(conv5_2)

    conv6_2 = Conv2D(filters=48, kerel_size=(5,5), dilation_rate=(8,1), stride=1)(conv5_2)
    conv6_2 = BatchNormalization()(conv6_2)
    conv6_2 = ReLU(conv6_2)

    conv7_2 = Conv2D(filters=48, kerel_size=(5,5), dilation_rate=(16,1), stride=1)(conv6_2)
    conv7_2 = BatchNormalization()(conv7_2)
    conv7_2 = ReLU(conv7_2)

    conv8_2 = Conv2D(filters=48, kerel_size=(5,5), dilation_rate=(32,1), stride=1)(conv7_2)
    conv8_2 = BatchNormalization()(conv8_2)
    conv8_2 = ReLU(conv8_2)

    conv9_2 = Conv2D(filters=48, kerel_size=(5,5), dilation_rate=(1,1), stride=1)(conv8_2)
    conv9_2 = BatchNormalization()(conv9_2)
    conv9_2 = ReLU(conv9_2)

    conv10_2 = Conv2D(filters=48, kerel_size=(5,5), dilation_rate=(2,2), stride=1)(conv9_2)
    conv10_2 = BatchNormalization()(conv10_2)
    conv10_2 = ReLU(conv10_2)

    conv11_2 = Conv2D(filters=48, kerel_size=(5,5), dilation_rate=(4,4), stride=1)(conv10_2)
    conv11_2 = BatchNormalization()(conv11_2)
    conv11_2 = ReLU(conv11_2)

    conv12_2 = Conv2D(filters=48, kerel_size=(5,5), dilation_rate=(8,8), stride=1)(conv11_2)
    conv12_2 = BatchNormalization()(conv12_2)
    conv12_2 = ReLU(conv12_2)

    conv13_2 = Conv2D(filters=48, kerel_size=(5,5), dilation_rate=(16,16), stride=1)(conv12_2)
    conv13_2 = BatchNormalization()(conv13_2)
    conv13_2 = ReLU(conv13_2)

    conv14_2 = Conv2D(filters=48, kerel_size=(5,5), dilation_rate=(32,32), stride=1)(conv13_2)
    conv14_2 = BatchNormalization()(conv14_2)
    conv14_2 = ReLU(conv14_2)

    conv15_2 = Conv2D(filters=4, kerel_size=(1,1), dilation_rate=(1,1), stride=1)(conv14_2)
    conv15_2 = BatchNormalization()(conv15_2)
    conv15_2 = ReLU(conv15_2)

    encoders_output = Concatenate([conv15_1, conv15_2])

    lstm = LSTM(units=200, activation="relu")(encoders_output)

    fc1 = Dense(units=600)(lstm)
    fc1 = ReLU(fc1)

    fc2 = Dense(units=600)(fc1)
    fc2 = ReLU(units=600)(fc2)

    nr_output = Dense(units=2*FREQUENCY_BINS, activation='sigmoid')(fc2)

    return nr_output



