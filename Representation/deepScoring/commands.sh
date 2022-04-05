python3 train_model_v2.py -c ~/save/conv_params/conv0.pkl --steps 20000 -t _95_2 --restore ~/save/tests_cluster/Model_95_1/
python3 evaluation.py -s gdt -d ~/data/CASP12_stage1/ -m ~/Temp/Model_95_2 -n _12stage_1
python3 evaluation.py -s cad -d ~/data/CASP12_stage1/ -m ~/Temp/Model_95_2 -n _12stage_1_cad
python3 evaluation.py -s gdt -d ~/data/CASP12_stage2/ -m ~/Temp/Model_95_2 -n _12stage_2
python3 evaluation.py -s cad -d ~/data/CASP12_stage2/ -m ~/Temp/Model_95_2 -n _12stage_2_cad
