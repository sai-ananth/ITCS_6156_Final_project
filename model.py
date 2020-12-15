from keras.models import Model
from keras.layers import Add, Activation, Concatenate, Conv2D, Dropout 
from keras.layers import Flatten, Input, GlobalAveragePooling2D, MaxPooling2D
import keras.backend as K
import random


def ModifiedSqueezenet(input_shape, nb_classes, dropout_rate=None, compression=1.0):
    input_img = Input(shape=input_shape)
   
    x = Conv2D(int(96*compression), (7,7), activation='relu', strides=(2,2), padding='same', name='conv1')(input_img)

    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='maxpool1')(x)
    
    x = create_mod_fire_module(x, int(64*compression), name='fire1')
    x = create_mod_fire_module(x, int(64*compression), name='fire2')
    
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='maxpool2')(x)
    
    x = create_mod_fire_module(x, int(32*compression), name='fire3')
    x = create_mod_fire_module(x, int(32*compression), name='fire4')
    
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='maxpool4')(x)
    
    x = create_mod_fire_module(x, int(16*compression), name='fire5')
    x = create_mod_fire_module(x, int(16*compression), name='fire6')
    
    
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
        
    x = Conv2D(nb_classes, (1,1), strides=(1,1), padding='valid', name='conv7')(x)
    x = GlobalAveragePooling2D(name='avgpool7')(x)
    x = Activation("softmax", name='softmax')(x)

    return Model(inputs=input_img, outputs=x)



def ModifiedSqueezenet_v2(input_shape, nb_classes, dropout_rate=None, compression=1.0,epsilon = 1):
    input_img = Input(shape=input_shape)
    
    prob = random.uniform(0, 1)
    taken = prob < epsilon
    x = Conv2D(int(96*compression), (7,7), activation='relu', strides=(2,2), padding='same', name='conv1')(input_img)

    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='maxpool1')(x)
    
    x = create_mod_fire_module(x, int(64*compression), name='fire1',taken = taken)
    x = create_mod_fire_module(x, int(64*compression), name='fire2',taken = not taken)
    
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='maxpool2')(x)
    
    x = create_mod_fire_module(x, int(32*compression), name='fire3',taken = taken)
    x = create_mod_fire_module(x, int(32*compression), name='fire4',taken = not taken)
    
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='maxpool4')(x)
    
    x = create_mod_fire_module(x, int(16*compression), name='fire5',taken = taken)
    x = create_mod_fire_module(x, int(16*compression), name='fire6',taken = not taken)
    
    
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
        
    x = Conv2D(nb_classes, (1,1), strides=(1,1), padding='valid', name='conv7')(x)
    x = GlobalAveragePooling2D(name='avgpool7')(x)
    x = Activation("softmax", name='softmax')(x)

    return Model(inputs=input_img, outputs=x)

def create_mod_fire_module(x, nb_squeeze_filter, name,taken = True):
    nb_expand_filter = 2 * nb_squeeze_filter
    squeeze    = Conv2D(nb_squeeze_filter,(1,1), activation='relu', padding='same', name='%s_squeeze'%name)(x)
    expand_1x1 = Conv2D(nb_expand_filter, (1,1), activation='relu', padding='same', name='%s_expand_1x1'%name)(squeeze)
    expand_3x3 = Conv2D(nb_expand_filter, (3,3), activation='relu', padding='same', name='%s_expand_3x3'%name)(squeeze)
    expand_5x5 = Conv2D(nb_expand_filter, (5,5), activation='relu', padding='same', name='%s_expand_5x5'%name)(squeeze)
   
    expand_maxpool = MaxPooling2D(pool_size=(3,3), strides=(1,1),padding='same', name='%s_expand_maxpool_3x3'%name)(squeeze)

    axis = -1 if K.image_data_format() == 'channels_last' else 1
    x_ret = Concatenate(axis=axis, name='%s_concatenate'%name)([expand_1x1, expand_3x3, expand_5x5, expand_maxpool])
    if(not taken):
        x_ret = Concatenate(axis=axis, name='%s_concatenate'%name)([expand_1x1, expand_3x3, expand_maxpool])
    

    return x_ret


