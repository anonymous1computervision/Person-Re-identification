'''
New RNN based model 
input shape:  (None, f, h, w, c)
f : no. of frames for each person sequence
h = 256
w = 128
c = 3
'''

from keras.layers import Conv2D, TimeDistributed, GlobalAveragePooling1D, Flatten, Dropout, LSTM, Subtract, Input, MaxPooling2D, Dense
import numpy as np
from keras.models import Model

class Rnn_reid:

    def build_model(self):
        
        #input shape:  (None, f, h, w, c)
        input1 = Input(shape=self.input_shape)
        print(input1.shape)
        x = TimeDistributed(Conv2D(32, (7, 7), strides=(2, 2), activation='relu', padding='same'))(input1)
        x = TimeDistributed(Conv2D(16, (3, 3), strides=(3, 3), activation='relu', padding='same'))(x)
        x = TimeDistributed(Conv2D(8, (3, 3), strides=(1, 1), activation='relu', padding='same'))(x)    
        x = TimeDistributed(MaxPooling2D((3, 3), strides=(3, 3)))(x)                
        x = TimeDistributed(Flatten())(x)                     #output: (None, f, 180)       
        #x = Dropout(0.5)(x)
        x = LSTM(256, return_sequences=True, dropout=0.5)(x)  #output: (None, f, 256)
        x = GlobalAveragePooling1D()(x)                       #output: (None, 256)       

        input2 = Input(shape=self.input_shape)
        y = TimeDistributed(Conv2D(32, (7, 7), strides=(2, 2), activation='relu', padding='same'))(input2)
        y = TimeDistributed(Conv2D(16, (3, 3), strides=(3, 3), activation='relu', padding='same'))(y)
        y = TimeDistributed(Conv2D(8, (3, 3), strides=(1, 1), activation='relu', padding='same'))(y)    
        y = TimeDistributed(MaxPooling2D((3, 3), strides=(3, 3)))(y)                
        y = TimeDistributed(Flatten())(y)   #output: (None, f, 180)
        #y = Dropout(0.5)(y)
        y = LSTM(256, return_sequences=True, dropout=0.5)(y)   #output: (None, f, 256)  
        y = GlobalAveragePooling1D()(y)                        #(None, 256)       

        out = Subtract()([x, y])
        out = Dense(2 , activation='sigmoid')(out)
        print(out.shape)
        model = Model( inputs=[input1,input2], outputs=out )

        return model
        
    def generator():

        folders = os.listdir( './bbox' )
        folder_index = 0
        while folder_index < len(folders):
            #each folder is a person
            folder1 = folders[folder_index]
            folder2 = folders[folder_index + 1]
            folder_index += 2;
            
            #feed pairs of similar sequences from folder1
            seq1 = []
            seq2 = []
            t1=1
            t2=2
            for img_name in os.listdir('./bbox/'+folder1):
                if( checkT(img_name) == t1 ):
                
            
            
        
    def __init__(self):
        f = 100
        h = 256
        w = 128
        c = 3
        self.input_shape = (f,h,w,c)
        self.model = self.build_model()
        print('\n\n')
        print(self.model.summary())
        self.model.compile(loss='binary_crossentropy', optimizer='adam')
        
        
if __name__ == '__main__':
     r = Rnn_reid() 



