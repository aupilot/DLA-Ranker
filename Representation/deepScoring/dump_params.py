import pickle
import model_v2_routed as m

arg = m.layer_params(['conv_20_3_VALID', 'conv_30_3_VALID', 'conv_40_3_VALID','conv_50_5_VALID','conv_60_5_VALID','conv_70_5_VALID','avgpool_2'])
with open('convdeep.pkl', 'wb') as output:
    pickle.dump(arg, output, pickle.HIGHEST_PROTOCOL)
    
arg = m.layer_params(['linear_512','linear_384','linear_256','linear_128','linear_64','linear_32','linear_16','linear_1'])
with open('fcdeep.pkl', 'wb') as output:
    pickle.dump(arg, output, pickle.HIGHEST_PROTOCOL)
    
arg = m.layer_params(['linear_512','linear_384','linear_256','linear_128','linear_64','linear_20'])
with open('fcdeep_router.pkl', 'wb') as output:
    pickle.dump(arg, output, pickle.HIGHEST_PROTOCOL)


arg = m.layer_params(['conv_20_3_VALID', 'conv_30_4_VALID', 'conv_20_4_VALID', 'avgpool_4'])
with open('/home/benoitch/save/conv_params/conv0.pkl', 'wb') as output:
    pickle.dump(arg, output, pickle.HIGHEST_PROTOCOL)

arg = m.layer_params(['conv_20_3_VALID', 'conv_40_4_VALID', 'conv_60_4_VALID', 'avgpool_4'])
with open('/home/benoitch/save/conv_params/conv1.pkl', 'wb') as output:
    pickle.dump(arg, output, pickle.HIGHEST_PROTOCOL)

arg = m.layer_params(['conv_20_3_VALID', 'conv_40_4_VALID', 'avgpool_2', 'conv_60_4_VALID', 'avgpool_2'])
with open('/home/benoitch/save/conv_params/conv2.pkl', 'wb') as output:
    pickle.dump(arg, output, pickle.HIGHEST_PROTOCOL)

arg = m.layer_params(['conv_20_5_VALID', 'conv_30_4_VALID', 'conv_30_4_VALID', 'conv_40_4_VALID', 'conv_40_4_VALID','avgpool_2'])
with open('/home/benoitch/save/conv_params/conv3.pkl', 'wb') as output:
    pickle.dump(arg, output, pickle.HIGHEST_PROTOCOL)

arg = m.layer_params(['conv_20_3_VALID', 'conv_30_4_VALID', 'conv_20_4_VALID','avgpool_4'])
with open('/home/benoitch/save/conv_params/conv4.pkl', 'wb') as output:
    pickle.dump(arg, output, pickle.HIGHEST_PROTOCOL)

arg = m.layer_params(['conv_20_3_VALID', 'conv_30_4_VALID', 'conv_20_4_VALID','avgpool_2'])
with open('/home/benoitch/save/conv_params/conv4.pkl', 'wb') as output:
    pickle.dump(arg, output, pickle.HIGHEST_PROTOCOL)





