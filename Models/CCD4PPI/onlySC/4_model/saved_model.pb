┬┴ 
ж¤
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
╛
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeИ"serve*2.2.02v2.2.0-0-g2b96f3662b8╣╖
З
conv3d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0* 
shape:н* 
shared_nameconv3d_4/kernel
А
#conv3d_4/kernel/Read/ReadVariableOpReadVariableOpconv3d_4/kernel*+
_output_shapes
:н*
dtype0
r
conv3d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_4/bias
k
!conv3d_4/bias/Read/ReadVariableOpReadVariableOpconv3d_4/bias*
_output_shapes
:*
dtype0
О
batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_4/gamma
З
/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
_output_shapes
:*
dtype0
М
batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_4/beta
Е
.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes
:*
dtype0
Ъ
!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_4/moving_mean
У
5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes
:*
dtype0
в
%batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_4/moving_variance
Ы
9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
_output_shapes
:*
dtype0
Ж
conv3d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv3d_5/kernel

#conv3d_5/kernel/Read/ReadVariableOpReadVariableOpconv3d_5/kernel**
_output_shapes
:*
dtype0
r
conv3d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_5/bias
k
!conv3d_5/bias/Read/ReadVariableOpReadVariableOpconv3d_5/bias*
_output_shapes
:*
dtype0
О
batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_5/gamma
З
/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_5/gamma*
_output_shapes
:*
dtype0
М
batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_5/beta
Е
.batch_normalization_5/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_5/beta*
_output_shapes
:*
dtype0
Ъ
!batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_5/moving_mean
У
5batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_5/moving_mean*
_output_shapes
:*
dtype0
в
%batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_5/moving_variance
Ы
9batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_5/moving_variance*
_output_shapes
:*
dtype0
Ж
conv3d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv3d_6/kernel

#conv3d_6/kernel/Read/ReadVariableOpReadVariableOpconv3d_6/kernel**
_output_shapes
:*
dtype0
r
conv3d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_6/bias
k
!conv3d_6/bias/Read/ReadVariableOpReadVariableOpconv3d_6/bias*
_output_shapes
:*
dtype0
О
batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_6/gamma
З
/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_6/gamma*
_output_shapes
:*
dtype0
М
batch_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_6/beta
Е
.batch_normalization_6/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_6/beta*
_output_shapes
:*
dtype0
Ъ
!batch_normalization_6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_6/moving_mean
У
5batch_normalization_6/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_6/moving_mean*
_output_shapes
:*
dtype0
в
%batch_normalization_6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_6/moving_variance
Ы
9batch_normalization_6/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_6/moving_variance*
_output_shapes
:*
dtype0
Ж
conv3d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv3d_7/kernel

#conv3d_7/kernel/Read/ReadVariableOpReadVariableOpconv3d_7/kernel**
_output_shapes
:*
dtype0
r
conv3d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv3d_7/bias
k
!conv3d_7/bias/Read/ReadVariableOpReadVariableOpconv3d_7/bias*
_output_shapes
:*
dtype0
О
batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_7/gamma
З
/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_7/gamma*
_output_shapes
:*
dtype0
М
batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_7/beta
Е
.batch_normalization_7/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_7/beta*
_output_shapes
:*
dtype0
Ъ
!batch_normalization_7/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_7/moving_mean
У
5batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_7/moving_mean*
_output_shapes
:*
dtype0
в
%batch_normalization_7/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_7/moving_variance
Ы
9batch_normalization_7/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_7/moving_variance*
_output_shapes
:*
dtype0
|
layer1_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Г
╚* 
shared_namelayer1_1/kernel
u
#layer1_1/kernel/Read/ReadVariableOpReadVariableOplayer1_1/kernel* 
_output_shapes
:
Г
╚*
dtype0
s
layer1_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:╚*
shared_namelayer1_1/bias
l
!layer1_1/bias/Read/ReadVariableOpReadVariableOplayer1_1/bias*
_output_shapes	
:╚*
dtype0
{
layer2_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	╚* 
shared_namelayer2_1/kernel
t
#layer2_1/kernel/Read/ReadVariableOpReadVariableOplayer2_1/kernel*
_output_shapes
:	╚*
dtype0
r
layer2_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namelayer2_1/bias
k
!layer2_1/bias/Read/ReadVariableOpReadVariableOplayer2_1/bias*
_output_shapes
:*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
Х
Adam/conv3d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0* 
shape:н*'
shared_nameAdam/conv3d_4/kernel/m
О
*Adam/conv3d_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_4/kernel/m*+
_output_shapes
:н*
dtype0
А
Adam/conv3d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_4/bias/m
y
(Adam/conv3d_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_4/bias/m*
_output_shapes
:*
dtype0
Ь
"Adam/batch_normalization_4/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_4/gamma/m
Х
6Adam/batch_normalization_4/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_4/gamma/m*
_output_shapes
:*
dtype0
Ъ
!Adam/batch_normalization_4/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_4/beta/m
У
5Adam/batch_normalization_4/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_4/beta/m*
_output_shapes
:*
dtype0
Ф
Adam/conv3d_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_5/kernel/m
Н
*Adam/conv3d_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_5/kernel/m**
_output_shapes
:*
dtype0
А
Adam/conv3d_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_5/bias/m
y
(Adam/conv3d_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_5/bias/m*
_output_shapes
:*
dtype0
Ь
"Adam/batch_normalization_5/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_5/gamma/m
Х
6Adam/batch_normalization_5/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_5/gamma/m*
_output_shapes
:*
dtype0
Ъ
!Adam/batch_normalization_5/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_5/beta/m
У
5Adam/batch_normalization_5/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_5/beta/m*
_output_shapes
:*
dtype0
Ф
Adam/conv3d_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_6/kernel/m
Н
*Adam/conv3d_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_6/kernel/m**
_output_shapes
:*
dtype0
А
Adam/conv3d_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_6/bias/m
y
(Adam/conv3d_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_6/bias/m*
_output_shapes
:*
dtype0
Ь
"Adam/batch_normalization_6/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_6/gamma/m
Х
6Adam/batch_normalization_6/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_6/gamma/m*
_output_shapes
:*
dtype0
Ъ
!Adam/batch_normalization_6/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_6/beta/m
У
5Adam/batch_normalization_6/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_6/beta/m*
_output_shapes
:*
dtype0
Ф
Adam/conv3d_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_7/kernel/m
Н
*Adam/conv3d_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_7/kernel/m**
_output_shapes
:*
dtype0
А
Adam/conv3d_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_7/bias/m
y
(Adam/conv3d_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3d_7/bias/m*
_output_shapes
:*
dtype0
Ь
"Adam/batch_normalization_7/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_7/gamma/m
Х
6Adam/batch_normalization_7/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_7/gamma/m*
_output_shapes
:*
dtype0
Ъ
!Adam/batch_normalization_7/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_7/beta/m
У
5Adam/batch_normalization_7/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_7/beta/m*
_output_shapes
:*
dtype0
К
Adam/layer1_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Г
╚*'
shared_nameAdam/layer1_1/kernel/m
Г
*Adam/layer1_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer1_1/kernel/m* 
_output_shapes
:
Г
╚*
dtype0
Б
Adam/layer1_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:╚*%
shared_nameAdam/layer1_1/bias/m
z
(Adam/layer1_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer1_1/bias/m*
_output_shapes	
:╚*
dtype0
Й
Adam/layer2_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	╚*'
shared_nameAdam/layer2_1/kernel/m
В
*Adam/layer2_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/layer2_1/kernel/m*
_output_shapes
:	╚*
dtype0
А
Adam/layer2_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/layer2_1/bias/m
y
(Adam/layer2_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/layer2_1/bias/m*
_output_shapes
:*
dtype0
Ж
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:*
dtype0
Х
Adam/conv3d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0* 
shape:н*'
shared_nameAdam/conv3d_4/kernel/v
О
*Adam/conv3d_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_4/kernel/v*+
_output_shapes
:н*
dtype0
А
Adam/conv3d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_4/bias/v
y
(Adam/conv3d_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_4/bias/v*
_output_shapes
:*
dtype0
Ь
"Adam/batch_normalization_4/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_4/gamma/v
Х
6Adam/batch_normalization_4/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_4/gamma/v*
_output_shapes
:*
dtype0
Ъ
!Adam/batch_normalization_4/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_4/beta/v
У
5Adam/batch_normalization_4/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_4/beta/v*
_output_shapes
:*
dtype0
Ф
Adam/conv3d_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_5/kernel/v
Н
*Adam/conv3d_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_5/kernel/v**
_output_shapes
:*
dtype0
А
Adam/conv3d_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_5/bias/v
y
(Adam/conv3d_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_5/bias/v*
_output_shapes
:*
dtype0
Ь
"Adam/batch_normalization_5/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_5/gamma/v
Х
6Adam/batch_normalization_5/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_5/gamma/v*
_output_shapes
:*
dtype0
Ъ
!Adam/batch_normalization_5/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_5/beta/v
У
5Adam/batch_normalization_5/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_5/beta/v*
_output_shapes
:*
dtype0
Ф
Adam/conv3d_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_6/kernel/v
Н
*Adam/conv3d_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_6/kernel/v**
_output_shapes
:*
dtype0
А
Adam/conv3d_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_6/bias/v
y
(Adam/conv3d_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_6/bias/v*
_output_shapes
:*
dtype0
Ь
"Adam/batch_normalization_6/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_6/gamma/v
Х
6Adam/batch_normalization_6/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_6/gamma/v*
_output_shapes
:*
dtype0
Ъ
!Adam/batch_normalization_6/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_6/beta/v
У
5Adam/batch_normalization_6/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_6/beta/v*
_output_shapes
:*
dtype0
Ф
Adam/conv3d_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv3d_7/kernel/v
Н
*Adam/conv3d_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_7/kernel/v**
_output_shapes
:*
dtype0
А
Adam/conv3d_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv3d_7/bias/v
y
(Adam/conv3d_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3d_7/bias/v*
_output_shapes
:*
dtype0
Ь
"Adam/batch_normalization_7/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_7/gamma/v
Х
6Adam/batch_normalization_7/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_7/gamma/v*
_output_shapes
:*
dtype0
Ъ
!Adam/batch_normalization_7/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_7/beta/v
У
5Adam/batch_normalization_7/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_7/beta/v*
_output_shapes
:*
dtype0
К
Adam/layer1_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Г
╚*'
shared_nameAdam/layer1_1/kernel/v
Г
*Adam/layer1_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer1_1/kernel/v* 
_output_shapes
:
Г
╚*
dtype0
Б
Adam/layer1_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:╚*%
shared_nameAdam/layer1_1/bias/v
z
(Adam/layer1_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer1_1/bias/v*
_output_shapes	
:╚*
dtype0
Й
Adam/layer2_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	╚*'
shared_nameAdam/layer2_1/kernel/v
В
*Adam/layer2_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/layer2_1/kernel/v*
_output_shapes
:	╚*
dtype0
А
Adam/layer2_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/layer2_1/bias/v
y
(Adam/layer2_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/layer2_1/bias/v*
_output_shapes
:*
dtype0
Ж
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
зЖ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*сЕ
value╓ЕB╥Е B╩Е
Р
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer_with_weights-8
layer-14
layer-15
layer_with_weights-9
layer-16
layer-17
layer_with_weights-10
layer-18
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
Ч
 axis
	!gamma
"beta
#moving_mean
$moving_variance
%trainable_variables
&regularization_losses
'	variables
(	keras_api
h

)kernel
*bias
+trainable_variables
,regularization_losses
-	variables
.	keras_api
Ч
/axis
	0gamma
1beta
2moving_mean
3moving_variance
4trainable_variables
5regularization_losses
6	variables
7	keras_api
h

8kernel
9bias
:trainable_variables
;regularization_losses
<	variables
=	keras_api
Ч
>axis
	?gamma
@beta
Amoving_mean
Bmoving_variance
Ctrainable_variables
Dregularization_losses
E	variables
F	keras_api
h

Gkernel
Hbias
Itrainable_variables
Jregularization_losses
K	variables
L	keras_api
Ч
Maxis
	Ngamma
Obeta
Pmoving_mean
Qmoving_variance
Rtrainable_variables
Sregularization_losses
T	variables
U	keras_api
R
Vtrainable_variables
Wregularization_losses
X	variables
Y	keras_api
R
Ztrainable_variables
[regularization_losses
\	variables
]	keras_api
R
^trainable_variables
_regularization_losses
`	variables
a	keras_api
 
R
btrainable_variables
cregularization_losses
d	variables
e	keras_api
h

fkernel
gbias
htrainable_variables
iregularization_losses
j	variables
k	keras_api
R
ltrainable_variables
mregularization_losses
n	variables
o	keras_api
h

pkernel
qbias
rtrainable_variables
sregularization_losses
t	variables
u	keras_api
R
vtrainable_variables
wregularization_losses
x	variables
y	keras_api
h

zkernel
{bias
|trainable_variables
}regularization_losses
~	variables
	keras_api
¤
	Аiter
Бbeta_1
Вbeta_2

Гdecay
Дlearning_ratemфmх!mц"mч)mш*mщ0mъ1mы8mь9mэ?mю@mяGmЁHmёNmЄOmєfmЇgmїpmЎqmўzm°{m∙v·v√!v№"v¤)v■*v 0vА1vБ8vВ9vГ?vД@vЕGvЖHvЗNvИOvЙfvКgvЛpvМqvНzvО{vП
ж
0
1
!2
"3
)4
*5
06
17
88
99
?10
@11
G12
H13
N14
O15
f16
g17
p18
q19
z20
{21
 
ц
0
1
!2
"3
#4
$5
)6
*7
08
19
210
311
812
913
?14
@15
A16
B17
G18
H19
N20
O21
P22
Q23
f24
g25
p26
q27
z28
{29
▓
Еlayer_metrics
Жnon_trainable_variables
Зlayers
Иmetrics
trainable_variables
regularization_losses
 Йlayer_regularization_losses
	variables
 
[Y
VARIABLE_VALUEconv3d_4/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_4/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
▓
Кlayer_metrics
Лnon_trainable_variables
Мmetrics
Нlayers
trainable_variables
regularization_losses
 Оlayer_regularization_losses
	variables
 
fd
VARIABLE_VALUEbatch_normalization_4/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_4/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_4/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_4/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

!0
"1
 

!0
"1
#2
$3
▓
Пlayer_metrics
Рnon_trainable_variables
Сmetrics
Тlayers
%trainable_variables
&regularization_losses
 Уlayer_regularization_losses
'	variables
[Y
VARIABLE_VALUEconv3d_5/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_5/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

)0
*1
 

)0
*1
▓
Фlayer_metrics
Хnon_trainable_variables
Цmetrics
Чlayers
+trainable_variables
,regularization_losses
 Шlayer_regularization_losses
-	variables
 
fd
VARIABLE_VALUEbatch_normalization_5/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_5/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_5/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_5/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

00
11
 

00
11
22
33
▓
Щlayer_metrics
Ъnon_trainable_variables
Ыmetrics
Ьlayers
4trainable_variables
5regularization_losses
 Эlayer_regularization_losses
6	variables
[Y
VARIABLE_VALUEconv3d_6/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_6/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

80
91
 

80
91
▓
Юlayer_metrics
Яnon_trainable_variables
аmetrics
бlayers
:trainable_variables
;regularization_losses
 вlayer_regularization_losses
<	variables
 
fd
VARIABLE_VALUEbatch_normalization_6/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_6/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_6/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_6/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

?0
@1
 

?0
@1
A2
B3
▓
гlayer_metrics
дnon_trainable_variables
еmetrics
жlayers
Ctrainable_variables
Dregularization_losses
 зlayer_regularization_losses
E	variables
[Y
VARIABLE_VALUEconv3d_7/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3d_7/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

G0
H1
 

G0
H1
▓
иlayer_metrics
йnon_trainable_variables
кmetrics
лlayers
Itrainable_variables
Jregularization_losses
 мlayer_regularization_losses
K	variables
 
fd
VARIABLE_VALUEbatch_normalization_7/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_7/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_7/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_7/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

N0
O1
 

N0
O1
P2
Q3
▓
нlayer_metrics
оnon_trainable_variables
пmetrics
░layers
Rtrainable_variables
Sregularization_losses
 ▒layer_regularization_losses
T	variables
 
 
 
▓
▓layer_metrics
│non_trainable_variables
┤metrics
╡layers
Vtrainable_variables
Wregularization_losses
 ╢layer_regularization_losses
X	variables
 
 
 
▓
╖layer_metrics
╕non_trainable_variables
╣metrics
║layers
Ztrainable_variables
[regularization_losses
 ╗layer_regularization_losses
\	variables
 
 
 
▓
╝layer_metrics
╜non_trainable_variables
╛metrics
┐layers
^trainable_variables
_regularization_losses
 └layer_regularization_losses
`	variables
 
 
 
▓
┴layer_metrics
┬non_trainable_variables
├metrics
─layers
btrainable_variables
cregularization_losses
 ┼layer_regularization_losses
d	variables
[Y
VARIABLE_VALUElayer1_1/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUElayer1_1/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

f0
g1
 

f0
g1
▓
╞layer_metrics
╟non_trainable_variables
╚metrics
╔layers
htrainable_variables
iregularization_losses
 ╩layer_regularization_losses
j	variables
 
 
 
▓
╦layer_metrics
╠non_trainable_variables
═metrics
╬layers
ltrainable_variables
mregularization_losses
 ╧layer_regularization_losses
n	variables
[Y
VARIABLE_VALUElayer2_1/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUElayer2_1/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

p0
q1
 

p0
q1
▓
╨layer_metrics
╤non_trainable_variables
╥metrics
╙layers
rtrainable_variables
sregularization_losses
 ╘layer_regularization_losses
t	variables
 
 
 
▓
╒layer_metrics
╓non_trainable_variables
╫metrics
╪layers
vtrainable_variables
wregularization_losses
 ┘layer_regularization_losses
x	variables
[Y
VARIABLE_VALUEdense_1/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_1/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

z0
{1
 

z0
{1
▓
┌layer_metrics
█non_trainable_variables
▄metrics
▌layers
|trainable_variables
}regularization_losses
 ▐layer_regularization_losses
~	variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
8
#0
$1
22
33
A4
B5
P6
Q7
О
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18

▀0
 
 
 
 
 
 
 

#0
$1
 
 
 
 
 
 
 
 
 

20
31
 
 
 
 
 
 
 
 
 

A0
B1
 
 
 
 
 
 
 
 
 

P0
Q1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

рtotal

сcount
т	variables
у	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

р0
с1

т	variables
~|
VARIABLE_VALUEAdam/conv3d_4/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_4/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_4/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE!Adam/batch_normalization_4/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_5/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_5/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_5/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE!Adam/batch_normalization_5/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_6/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_6/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_6/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE!Adam/batch_normalization_6/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_7/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_7/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_7/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE!Adam/batch_normalization_7/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/layer1_1/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/layer1_1/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/layer2_1/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/layer2_1/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_1/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_1/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_4/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_4/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_4/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE!Adam/batch_normalization_4/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_5/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_5/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_5/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE!Adam/batch_normalization_5/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_6/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_6/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_6/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE!Adam/batch_normalization_6/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv3d_7/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3d_7/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_7/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE!Adam/batch_normalization_7/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/layer1_1/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/layer1_1/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/layer2_1/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/layer2_1/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_1/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_1/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ф
serving_default_input_3Placeholder*4
_output_shapes"
 :         н*
dtype0*)
shape :         н
z
serving_default_input_4Placeholder*'
_output_shapes
:         *
dtype0*
shape:         
╒
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_3serving_default_input_4conv3d_4/kernelconv3d_4/bias%batch_normalization_4/moving_variancebatch_normalization_4/gamma!batch_normalization_4/moving_meanbatch_normalization_4/betaconv3d_5/kernelconv3d_5/bias%batch_normalization_5/moving_variancebatch_normalization_5/gamma!batch_normalization_5/moving_meanbatch_normalization_5/betaconv3d_6/kernelconv3d_6/bias%batch_normalization_6/moving_variancebatch_normalization_6/gamma!batch_normalization_6/moving_meanbatch_normalization_6/betaconv3d_7/kernelconv3d_7/bias%batch_normalization_7/moving_variancebatch_normalization_7/gamma!batch_normalization_7/moving_meanbatch_normalization_7/betalayer1_1/kernellayer1_1/biaslayer2_1/kernellayer2_1/biasdense_1/kerneldense_1/bias*+
Tin$
"2 *
Tout
2*'
_output_shapes
:         *@
_read_only_resource_inputs"
 	
*-
config_proto

CPU

GPU2*0J 8*/
f*R(
&__inference_signature_wrapper_23020833
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ш
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv3d_4/kernel/Read/ReadVariableOp!conv3d_4/bias/Read/ReadVariableOp/batch_normalization_4/gamma/Read/ReadVariableOp.batch_normalization_4/beta/Read/ReadVariableOp5batch_normalization_4/moving_mean/Read/ReadVariableOp9batch_normalization_4/moving_variance/Read/ReadVariableOp#conv3d_5/kernel/Read/ReadVariableOp!conv3d_5/bias/Read/ReadVariableOp/batch_normalization_5/gamma/Read/ReadVariableOp.batch_normalization_5/beta/Read/ReadVariableOp5batch_normalization_5/moving_mean/Read/ReadVariableOp9batch_normalization_5/moving_variance/Read/ReadVariableOp#conv3d_6/kernel/Read/ReadVariableOp!conv3d_6/bias/Read/ReadVariableOp/batch_normalization_6/gamma/Read/ReadVariableOp.batch_normalization_6/beta/Read/ReadVariableOp5batch_normalization_6/moving_mean/Read/ReadVariableOp9batch_normalization_6/moving_variance/Read/ReadVariableOp#conv3d_7/kernel/Read/ReadVariableOp!conv3d_7/bias/Read/ReadVariableOp/batch_normalization_7/gamma/Read/ReadVariableOp.batch_normalization_7/beta/Read/ReadVariableOp5batch_normalization_7/moving_mean/Read/ReadVariableOp9batch_normalization_7/moving_variance/Read/ReadVariableOp#layer1_1/kernel/Read/ReadVariableOp!layer1_1/bias/Read/ReadVariableOp#layer2_1/kernel/Read/ReadVariableOp!layer2_1/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/conv3d_4/kernel/m/Read/ReadVariableOp(Adam/conv3d_4/bias/m/Read/ReadVariableOp6Adam/batch_normalization_4/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_4/beta/m/Read/ReadVariableOp*Adam/conv3d_5/kernel/m/Read/ReadVariableOp(Adam/conv3d_5/bias/m/Read/ReadVariableOp6Adam/batch_normalization_5/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_5/beta/m/Read/ReadVariableOp*Adam/conv3d_6/kernel/m/Read/ReadVariableOp(Adam/conv3d_6/bias/m/Read/ReadVariableOp6Adam/batch_normalization_6/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_6/beta/m/Read/ReadVariableOp*Adam/conv3d_7/kernel/m/Read/ReadVariableOp(Adam/conv3d_7/bias/m/Read/ReadVariableOp6Adam/batch_normalization_7/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_7/beta/m/Read/ReadVariableOp*Adam/layer1_1/kernel/m/Read/ReadVariableOp(Adam/layer1_1/bias/m/Read/ReadVariableOp*Adam/layer2_1/kernel/m/Read/ReadVariableOp(Adam/layer2_1/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp*Adam/conv3d_4/kernel/v/Read/ReadVariableOp(Adam/conv3d_4/bias/v/Read/ReadVariableOp6Adam/batch_normalization_4/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_4/beta/v/Read/ReadVariableOp*Adam/conv3d_5/kernel/v/Read/ReadVariableOp(Adam/conv3d_5/bias/v/Read/ReadVariableOp6Adam/batch_normalization_5/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_5/beta/v/Read/ReadVariableOp*Adam/conv3d_6/kernel/v/Read/ReadVariableOp(Adam/conv3d_6/bias/v/Read/ReadVariableOp6Adam/batch_normalization_6/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_6/beta/v/Read/ReadVariableOp*Adam/conv3d_7/kernel/v/Read/ReadVariableOp(Adam/conv3d_7/bias/v/Read/ReadVariableOp6Adam/batch_normalization_7/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_7/beta/v/Read/ReadVariableOp*Adam/layer1_1/kernel/v/Read/ReadVariableOp(Adam/layer1_1/bias/v/Read/ReadVariableOp*Adam/layer2_1/kernel/v/Read/ReadVariableOp(Adam/layer2_1/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOpConst*^
TinW
U2S	*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8**
f%R#
!__inference__traced_save_23022536
┐
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv3d_4/kernelconv3d_4/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_varianceconv3d_5/kernelconv3d_5/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_varianceconv3d_6/kernelconv3d_6/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_varianceconv3d_7/kernelconv3d_7/biasbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_variancelayer1_1/kernellayer1_1/biaslayer2_1/kernellayer2_1/biasdense_1/kerneldense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/conv3d_4/kernel/mAdam/conv3d_4/bias/m"Adam/batch_normalization_4/gamma/m!Adam/batch_normalization_4/beta/mAdam/conv3d_5/kernel/mAdam/conv3d_5/bias/m"Adam/batch_normalization_5/gamma/m!Adam/batch_normalization_5/beta/mAdam/conv3d_6/kernel/mAdam/conv3d_6/bias/m"Adam/batch_normalization_6/gamma/m!Adam/batch_normalization_6/beta/mAdam/conv3d_7/kernel/mAdam/conv3d_7/bias/m"Adam/batch_normalization_7/gamma/m!Adam/batch_normalization_7/beta/mAdam/layer1_1/kernel/mAdam/layer1_1/bias/mAdam/layer2_1/kernel/mAdam/layer2_1/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/conv3d_4/kernel/vAdam/conv3d_4/bias/v"Adam/batch_normalization_4/gamma/v!Adam/batch_normalization_4/beta/vAdam/conv3d_5/kernel/vAdam/conv3d_5/bias/v"Adam/batch_normalization_5/gamma/v!Adam/batch_normalization_5/beta/vAdam/conv3d_6/kernel/vAdam/conv3d_6/bias/v"Adam/batch_normalization_6/gamma/v!Adam/batch_normalization_6/beta/vAdam/conv3d_7/kernel/vAdam/conv3d_7/bias/v"Adam/batch_normalization_7/gamma/v!Adam/batch_normalization_7/beta/vAdam/layer1_1/kernel/vAdam/layer1_1/bias/vAdam/layer2_1/kernel/vAdam/layer2_1/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/v*]
TinV
T2R*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*-
f(R&
$__inference__traced_restore_23022791║╟
ъ
м
D__inference_layer1_layer_call_and_return_conditional_losses_23020103

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Г
╚*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2	
BiasAddV
EluEluBiasAdd:output:0*
T0*(
_output_shapes
:         ╚2
Eluf
IdentityIdentityElu:activations:0*
T0*(
_output_shapes
:         ╚2

Identity"
identityIdentity:output:0*/
_input_shapes
:         Г
:::P L
(
_output_shapes
:         Г

 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
щ,
╨
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_23019586

inputs
assignmovingavg_23019561
assignmovingavg_1_23019567)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpЩ
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indicesЫ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/meanИ
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient╦
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*N
_output_shapes<
::8                                    2
moments/SquaredDifferenceб
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices╛
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/varianceГ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЛ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1а
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/23019561*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayХ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_23019561*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp┼
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/23019561*
_output_shapes
:2
AssignMovingAvg/sub╝
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/23019561*
_output_shapes
:2
AssignMovingAvg/mulЕ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_23019561AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/23019561*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpж
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/23019567*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЫ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_23019567*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╧
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/23019567*
_output_shapes
:2
AssignMovingAvg_1/sub╞
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/23019567*
_output_shapes
:2
AssignMovingAvg_1/mulС
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_23019567AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/23019567*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulЭ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subм
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/add_1▄
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8                                    ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
┬
Ц
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_23021592

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulЭ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subм
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/add_1О
IdentityIdentitybatchnorm/add_1:z:0*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8                                    :::::v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
╟+
╨
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_23019879

inputs
assignmovingavg_23019854
assignmovingavg_1_23019860)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpЩ
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indicesЫ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/meanИ
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient░
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*3
_output_shapes!
:         2
moments/SquaredDifferenceб
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices╛
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/varianceГ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЛ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1а
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/23019854*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayХ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_23019854*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp┼
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/23019854*
_output_shapes
:2
AssignMovingAvg/sub╝
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/23019854*
_output_shapes
:2
AssignMovingAvg/mulЕ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_23019854AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/23019854*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpж
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/23019860*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЫ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_23019860*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╧
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/23019860*
_output_shapes
:2
AssignMovingAvg_1/sub╞
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/23019860*
_output_shapes
:2
AssignMovingAvg_1/mulС
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_23019860AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/23019860*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulВ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:         2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subС
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2
batchnorm/add_1┴
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:         ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Г
f
G__inference_dropout_5_layer_call_and_return_conditional_losses_23022230

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
dropout/GreaterEqual/y╛
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
їЮ
╫
E__inference_model_1_layer_call_and_return_conditional_losses_23021168
inputs_0
inputs_1+
'conv3d_4_conv3d_readvariableop_resource,
(conv3d_4_biasadd_readvariableop_resource;
7batch_normalization_4_batchnorm_readvariableop_resource?
;batch_normalization_4_batchnorm_mul_readvariableop_resource=
9batch_normalization_4_batchnorm_readvariableop_1_resource=
9batch_normalization_4_batchnorm_readvariableop_2_resource+
'conv3d_5_conv3d_readvariableop_resource,
(conv3d_5_biasadd_readvariableop_resource;
7batch_normalization_5_batchnorm_readvariableop_resource?
;batch_normalization_5_batchnorm_mul_readvariableop_resource=
9batch_normalization_5_batchnorm_readvariableop_1_resource=
9batch_normalization_5_batchnorm_readvariableop_2_resource+
'conv3d_6_conv3d_readvariableop_resource,
(conv3d_6_biasadd_readvariableop_resource;
7batch_normalization_6_batchnorm_readvariableop_resource?
;batch_normalization_6_batchnorm_mul_readvariableop_resource=
9batch_normalization_6_batchnorm_readvariableop_1_resource=
9batch_normalization_6_batchnorm_readvariableop_2_resource+
'conv3d_7_conv3d_readvariableop_resource,
(conv3d_7_biasadd_readvariableop_resource;
7batch_normalization_7_batchnorm_readvariableop_resource?
;batch_normalization_7_batchnorm_mul_readvariableop_resource=
9batch_normalization_7_batchnorm_readvariableop_1_resource=
9batch_normalization_7_batchnorm_readvariableop_2_resource)
%layer1_matmul_readvariableop_resource*
&layer1_biasadd_readvariableop_resource)
%layer2_matmul_readvariableop_resource*
&layer2_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identityИ╡
conv3d_4/Conv3D/ReadVariableOpReadVariableOp'conv3d_4_conv3d_readvariableop_resource*+
_output_shapes
:н*
dtype02 
conv3d_4/Conv3D/ReadVariableOp╞
conv3d_4/Conv3DConv3Dinputs_0&conv3d_4/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         *
paddingVALID*
strides	
2
conv3d_4/Conv3Dз
conv3d_4/BiasAdd/ReadVariableOpReadVariableOp(conv3d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv3d_4/BiasAdd/ReadVariableOp░
conv3d_4/BiasAddBiasAddconv3d_4/Conv3D:output:0'conv3d_4/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         2
conv3d_4/BiasAdd╘
.batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype020
.batch_normalization_4/batchnorm/ReadVariableOpУ
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_4/batchnorm/add/yр
#batch_normalization_4/batchnorm/addAddV26batch_normalization_4/batchnorm/ReadVariableOp:value:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_4/batchnorm/addе
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_4/batchnorm/Rsqrtр
2batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization_4/batchnorm/mul/ReadVariableOp▌
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:0:batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_4/batchnorm/mul╫
%batch_normalization_4/batchnorm/mul_1Mulconv3d_4/BiasAdd:output:0'batch_normalization_4/batchnorm/mul:z:0*
T0*3
_output_shapes!
:         2'
%batch_normalization_4/batchnorm/mul_1┌
0batch_normalization_4/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_4_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype022
0batch_normalization_4/batchnorm/ReadVariableOp_1▌
%batch_normalization_4/batchnorm/mul_2Mul8batch_normalization_4/batchnorm/ReadVariableOp_1:value:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_4/batchnorm/mul_2┌
0batch_normalization_4/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_4_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype022
0batch_normalization_4/batchnorm/ReadVariableOp_2█
#batch_normalization_4/batchnorm/subSub8batch_normalization_4/batchnorm/ReadVariableOp_2:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_4/batchnorm/subщ
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2'
%batch_normalization_4/batchnorm/add_1┤
conv3d_5/Conv3D/ReadVariableOpReadVariableOp'conv3d_5_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02 
conv3d_5/Conv3D/ReadVariableOpч
conv3d_5/Conv3DConv3D)batch_normalization_4/batchnorm/add_1:z:0&conv3d_5/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         *
paddingVALID*
strides	
2
conv3d_5/Conv3Dз
conv3d_5/BiasAdd/ReadVariableOpReadVariableOp(conv3d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv3d_5/BiasAdd/ReadVariableOp░
conv3d_5/BiasAddBiasAddconv3d_5/Conv3D:output:0'conv3d_5/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         2
conv3d_5/BiasAdd|
conv3d_5/EluEluconv3d_5/BiasAdd:output:0*
T0*3
_output_shapes!
:         2
conv3d_5/Elu╘
.batch_normalization_5/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype020
.batch_normalization_5/batchnorm/ReadVariableOpУ
%batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_5/batchnorm/add/yр
#batch_normalization_5/batchnorm/addAddV26batch_normalization_5/batchnorm/ReadVariableOp:value:0.batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_5/batchnorm/addе
%batch_normalization_5/batchnorm/RsqrtRsqrt'batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_5/batchnorm/Rsqrtр
2batch_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization_5/batchnorm/mul/ReadVariableOp▌
#batch_normalization_5/batchnorm/mulMul)batch_normalization_5/batchnorm/Rsqrt:y:0:batch_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_5/batchnorm/mul╪
%batch_normalization_5/batchnorm/mul_1Mulconv3d_5/Elu:activations:0'batch_normalization_5/batchnorm/mul:z:0*
T0*3
_output_shapes!
:         2'
%batch_normalization_5/batchnorm/mul_1┌
0batch_normalization_5/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_5_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype022
0batch_normalization_5/batchnorm/ReadVariableOp_1▌
%batch_normalization_5/batchnorm/mul_2Mul8batch_normalization_5/batchnorm/ReadVariableOp_1:value:0'batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_5/batchnorm/mul_2┌
0batch_normalization_5/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_5_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype022
0batch_normalization_5/batchnorm/ReadVariableOp_2█
#batch_normalization_5/batchnorm/subSub8batch_normalization_5/batchnorm/ReadVariableOp_2:value:0)batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_5/batchnorm/subщ
%batch_normalization_5/batchnorm/add_1AddV2)batch_normalization_5/batchnorm/mul_1:z:0'batch_normalization_5/batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2'
%batch_normalization_5/batchnorm/add_1┤
conv3d_6/Conv3D/ReadVariableOpReadVariableOp'conv3d_6_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02 
conv3d_6/Conv3D/ReadVariableOpч
conv3d_6/Conv3DConv3D)batch_normalization_5/batchnorm/add_1:z:0&conv3d_6/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         *
paddingVALID*
strides	
2
conv3d_6/Conv3Dз
conv3d_6/BiasAdd/ReadVariableOpReadVariableOp(conv3d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv3d_6/BiasAdd/ReadVariableOp░
conv3d_6/BiasAddBiasAddconv3d_6/Conv3D:output:0'conv3d_6/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         2
conv3d_6/BiasAdd|
conv3d_6/EluEluconv3d_6/BiasAdd:output:0*
T0*3
_output_shapes!
:         2
conv3d_6/Elu╘
.batch_normalization_6/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_6_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype020
.batch_normalization_6/batchnorm/ReadVariableOpУ
%batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_6/batchnorm/add/yр
#batch_normalization_6/batchnorm/addAddV26batch_normalization_6/batchnorm/ReadVariableOp:value:0.batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_6/batchnorm/addе
%batch_normalization_6/batchnorm/RsqrtRsqrt'batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_6/batchnorm/Rsqrtр
2batch_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization_6/batchnorm/mul/ReadVariableOp▌
#batch_normalization_6/batchnorm/mulMul)batch_normalization_6/batchnorm/Rsqrt:y:0:batch_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_6/batchnorm/mul╪
%batch_normalization_6/batchnorm/mul_1Mulconv3d_6/Elu:activations:0'batch_normalization_6/batchnorm/mul:z:0*
T0*3
_output_shapes!
:         2'
%batch_normalization_6/batchnorm/mul_1┌
0batch_normalization_6/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_6_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype022
0batch_normalization_6/batchnorm/ReadVariableOp_1▌
%batch_normalization_6/batchnorm/mul_2Mul8batch_normalization_6/batchnorm/ReadVariableOp_1:value:0'batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_6/batchnorm/mul_2┌
0batch_normalization_6/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_6_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype022
0batch_normalization_6/batchnorm/ReadVariableOp_2█
#batch_normalization_6/batchnorm/subSub8batch_normalization_6/batchnorm/ReadVariableOp_2:value:0)batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_6/batchnorm/subщ
%batch_normalization_6/batchnorm/add_1AddV2)batch_normalization_6/batchnorm/mul_1:z:0'batch_normalization_6/batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2'
%batch_normalization_6/batchnorm/add_1┤
conv3d_7/Conv3D/ReadVariableOpReadVariableOp'conv3d_7_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02 
conv3d_7/Conv3D/ReadVariableOpч
conv3d_7/Conv3DConv3D)batch_normalization_6/batchnorm/add_1:z:0&conv3d_7/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         *
paddingVALID*
strides	
2
conv3d_7/Conv3Dз
conv3d_7/BiasAdd/ReadVariableOpReadVariableOp(conv3d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv3d_7/BiasAdd/ReadVariableOp░
conv3d_7/BiasAddBiasAddconv3d_7/Conv3D:output:0'conv3d_7/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         2
conv3d_7/BiasAdd|
conv3d_7/EluEluconv3d_7/BiasAdd:output:0*
T0*3
_output_shapes!
:         2
conv3d_7/Elu╘
.batch_normalization_7/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype020
.batch_normalization_7/batchnorm/ReadVariableOpУ
%batch_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_7/batchnorm/add/yр
#batch_normalization_7/batchnorm/addAddV26batch_normalization_7/batchnorm/ReadVariableOp:value:0.batch_normalization_7/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_7/batchnorm/addе
%batch_normalization_7/batchnorm/RsqrtRsqrt'batch_normalization_7/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_7/batchnorm/Rsqrtр
2batch_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization_7/batchnorm/mul/ReadVariableOp▌
#batch_normalization_7/batchnorm/mulMul)batch_normalization_7/batchnorm/Rsqrt:y:0:batch_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_7/batchnorm/mul╪
%batch_normalization_7/batchnorm/mul_1Mulconv3d_7/Elu:activations:0'batch_normalization_7/batchnorm/mul:z:0*
T0*3
_output_shapes!
:         2'
%batch_normalization_7/batchnorm/mul_1┌
0batch_normalization_7/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_7_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype022
0batch_normalization_7/batchnorm/ReadVariableOp_1▌
%batch_normalization_7/batchnorm/mul_2Mul8batch_normalization_7/batchnorm/ReadVariableOp_1:value:0'batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_7/batchnorm/mul_2┌
0batch_normalization_7/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_7_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype022
0batch_normalization_7/batchnorm/ReadVariableOp_2█
#batch_normalization_7/batchnorm/subSub8batch_normalization_7/batchnorm/ReadVariableOp_2:value:0)batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_7/batchnorm/subщ
%batch_normalization_7/batchnorm/add_1AddV2)batch_normalization_7/batchnorm/mul_1:z:0'batch_normalization_7/batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2'
%batch_normalization_7/batchnorm/add_1Є
average_pooling3d_1/AvgPool3D	AvgPool3D)batch_normalization_7/batchnorm/add_1:z:0*
T0*3
_output_shapes!
:         *
ksize	
*
paddingVALID*
strides	
2
average_pooling3d_1/AvgPool3Ds
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
flatten_1/Constж
flatten_1/ReshapeReshape&average_pooling3d_1/AvgPool3D:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:         А
2
flatten_1/ReshapeГ
dropout_3/IdentityIdentityflatten_1/Reshape:output:0*
T0*(
_output_shapes
:         А
2
dropout_3/Identityx
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis┐
concatenate_1/concatConcatV2dropout_3/Identity:output:0inputs_1"concatenate_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:         Г
2
concatenate_1/concatд
layer1/MatMul/ReadVariableOpReadVariableOp%layer1_matmul_readvariableop_resource* 
_output_shapes
:
Г
╚*
dtype02
layer1/MatMul/ReadVariableOpа
layer1/MatMulMatMulconcatenate_1/concat:output:0$layer1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
layer1/MatMulв
layer1/BiasAdd/ReadVariableOpReadVariableOp&layer1_biasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype02
layer1/BiasAdd/ReadVariableOpЮ
layer1/BiasAddBiasAddlayer1/MatMul:product:0%layer1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
layer1/BiasAddk

layer1/EluElulayer1/BiasAdd:output:0*
T0*(
_output_shapes
:         ╚2

layer1/EluБ
dropout_4/IdentityIdentitylayer1/Elu:activations:0*
T0*(
_output_shapes
:         ╚2
dropout_4/Identityг
layer2/MatMul/ReadVariableOpReadVariableOp%layer2_matmul_readvariableop_resource*
_output_shapes
:	╚*
dtype02
layer2/MatMul/ReadVariableOpЭ
layer2/MatMulMatMuldropout_4/Identity:output:0$layer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
layer2/MatMulб
layer2/BiasAdd/ReadVariableOpReadVariableOp&layer2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
layer2/BiasAdd/ReadVariableOpЭ
layer2/BiasAddBiasAddlayer2/MatMul:product:0%layer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
layer2/BiasAddj

layer2/EluElulayer2/BiasAdd:output:0*
T0*'
_output_shapes
:         2

layer2/EluА
dropout_5/IdentityIdentitylayer2/Elu:activations:0*
T0*'
_output_shapes
:         2
dropout_5/Identityе
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_1/MatMul/ReadVariableOpа
dense_1/MatMulMatMuldropout_5/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_1/MatMulд
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOpб
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_1/BiasAddy
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_1/Sigmoidg
IdentityIdentitydense_1/Sigmoid:y:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*└
_input_shapesо
л:         н:         :::::::::::::::::::::::::::::::^ Z
4
_output_shapes"
 :         н
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
√

*__inference_dense_1_layer_call_fn_23022265

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall╓
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_230202172
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
║
Ц
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_23019995

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulВ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:         2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subС
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:         :::::[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ф
м
D__inference_layer2_layer_call_and_return_conditional_losses_23020160

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	╚*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddU
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:         2
Elue
IdentityIdentityElu:activations:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ╚:::P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
щ,
╨
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_23021372

inputs
assignmovingavg_23021347
assignmovingavg_1_23021353)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpЩ
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indicesЫ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/meanИ
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient╦
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*N
_output_shapes<
::8                                    2
moments/SquaredDifferenceб
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices╛
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/varianceГ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЛ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1а
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/23021347*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayХ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_23021347*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp┼
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/23021347*
_output_shapes
:2
AssignMovingAvg/sub╝
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/23021347*
_output_shapes
:2
AssignMovingAvg/mulЕ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_23021347AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/23021347*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpж
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/23021353*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЫ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_23021353*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╧
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/23021353*
_output_shapes
:2
AssignMovingAvg_1/sub╞
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/23021353*
_output_shapes
:2
AssignMovingAvg_1/mulС
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_23021353AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/23021353*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulЭ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subм
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/add_1▄
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8                                    ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Д
о
F__inference_conv3d_7_layer_call_and_return_conditional_losses_23019480

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identityИЩ
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype02
Conv3D/ReadVariableOp─
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8                                    *
paddingVALID*
strides	
2
Conv3DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpз
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8                                    2	
BiasAdd|
EluEluBiasAdd:output:0*
T0*N
_output_shapes<
::8                                    2
EluМ
IdentityIdentityElu:activations:0*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8                                    :::v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
¤
H
,__inference_dropout_4_layer_call_fn_23022198

inputs
identityз
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:         ╚* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_dropout_4_layer_call_and_return_conditional_losses_230201362
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ╚2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ╚:P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs
л▌
╨-
$__inference__traced_restore_23022791
file_prefix$
 assignvariableop_conv3d_4_kernel$
 assignvariableop_1_conv3d_4_bias2
.assignvariableop_2_batch_normalization_4_gamma1
-assignvariableop_3_batch_normalization_4_beta8
4assignvariableop_4_batch_normalization_4_moving_mean<
8assignvariableop_5_batch_normalization_4_moving_variance&
"assignvariableop_6_conv3d_5_kernel$
 assignvariableop_7_conv3d_5_bias2
.assignvariableop_8_batch_normalization_5_gamma1
-assignvariableop_9_batch_normalization_5_beta9
5assignvariableop_10_batch_normalization_5_moving_mean=
9assignvariableop_11_batch_normalization_5_moving_variance'
#assignvariableop_12_conv3d_6_kernel%
!assignvariableop_13_conv3d_6_bias3
/assignvariableop_14_batch_normalization_6_gamma2
.assignvariableop_15_batch_normalization_6_beta9
5assignvariableop_16_batch_normalization_6_moving_mean=
9assignvariableop_17_batch_normalization_6_moving_variance'
#assignvariableop_18_conv3d_7_kernel%
!assignvariableop_19_conv3d_7_bias3
/assignvariableop_20_batch_normalization_7_gamma2
.assignvariableop_21_batch_normalization_7_beta9
5assignvariableop_22_batch_normalization_7_moving_mean=
9assignvariableop_23_batch_normalization_7_moving_variance'
#assignvariableop_24_layer1_1_kernel%
!assignvariableop_25_layer1_1_bias'
#assignvariableop_26_layer2_1_kernel%
!assignvariableop_27_layer2_1_bias&
"assignvariableop_28_dense_1_kernel$
 assignvariableop_29_dense_1_bias!
assignvariableop_30_adam_iter#
assignvariableop_31_adam_beta_1#
assignvariableop_32_adam_beta_2"
assignvariableop_33_adam_decay*
&assignvariableop_34_adam_learning_rate
assignvariableop_35_total
assignvariableop_36_count.
*assignvariableop_37_adam_conv3d_4_kernel_m,
(assignvariableop_38_adam_conv3d_4_bias_m:
6assignvariableop_39_adam_batch_normalization_4_gamma_m9
5assignvariableop_40_adam_batch_normalization_4_beta_m.
*assignvariableop_41_adam_conv3d_5_kernel_m,
(assignvariableop_42_adam_conv3d_5_bias_m:
6assignvariableop_43_adam_batch_normalization_5_gamma_m9
5assignvariableop_44_adam_batch_normalization_5_beta_m.
*assignvariableop_45_adam_conv3d_6_kernel_m,
(assignvariableop_46_adam_conv3d_6_bias_m:
6assignvariableop_47_adam_batch_normalization_6_gamma_m9
5assignvariableop_48_adam_batch_normalization_6_beta_m.
*assignvariableop_49_adam_conv3d_7_kernel_m,
(assignvariableop_50_adam_conv3d_7_bias_m:
6assignvariableop_51_adam_batch_normalization_7_gamma_m9
5assignvariableop_52_adam_batch_normalization_7_beta_m.
*assignvariableop_53_adam_layer1_1_kernel_m,
(assignvariableop_54_adam_layer1_1_bias_m.
*assignvariableop_55_adam_layer2_1_kernel_m,
(assignvariableop_56_adam_layer2_1_bias_m-
)assignvariableop_57_adam_dense_1_kernel_m+
'assignvariableop_58_adam_dense_1_bias_m.
*assignvariableop_59_adam_conv3d_4_kernel_v,
(assignvariableop_60_adam_conv3d_4_bias_v:
6assignvariableop_61_adam_batch_normalization_4_gamma_v9
5assignvariableop_62_adam_batch_normalization_4_beta_v.
*assignvariableop_63_adam_conv3d_5_kernel_v,
(assignvariableop_64_adam_conv3d_5_bias_v:
6assignvariableop_65_adam_batch_normalization_5_gamma_v9
5assignvariableop_66_adam_batch_normalization_5_beta_v.
*assignvariableop_67_adam_conv3d_6_kernel_v,
(assignvariableop_68_adam_conv3d_6_bias_v:
6assignvariableop_69_adam_batch_normalization_6_gamma_v9
5assignvariableop_70_adam_batch_normalization_6_beta_v.
*assignvariableop_71_adam_conv3d_7_kernel_v,
(assignvariableop_72_adam_conv3d_7_bias_v:
6assignvariableop_73_adam_batch_normalization_7_gamma_v9
5assignvariableop_74_adam_batch_normalization_7_beta_v.
*assignvariableop_75_adam_layer1_1_kernel_v,
(assignvariableop_76_adam_layer1_1_bias_v.
*assignvariableop_77_adam_layer2_1_kernel_v,
(assignvariableop_78_adam_layer2_1_bias_v-
)assignvariableop_79_adam_dense_1_kernel_v+
'assignvariableop_80_adam_dense_1_bias_v
identity_82ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_39вAssignVariableOp_4вAssignVariableOp_40вAssignVariableOp_41вAssignVariableOp_42вAssignVariableOp_43вAssignVariableOp_44вAssignVariableOp_45вAssignVariableOp_46вAssignVariableOp_47вAssignVariableOp_48вAssignVariableOp_49вAssignVariableOp_5вAssignVariableOp_50вAssignVariableOp_51вAssignVariableOp_52вAssignVariableOp_53вAssignVariableOp_54вAssignVariableOp_55вAssignVariableOp_56вAssignVariableOp_57вAssignVariableOp_58вAssignVariableOp_59вAssignVariableOp_6вAssignVariableOp_60вAssignVariableOp_61вAssignVariableOp_62вAssignVariableOp_63вAssignVariableOp_64вAssignVariableOp_65вAssignVariableOp_66вAssignVariableOp_67вAssignVariableOp_68вAssignVariableOp_69вAssignVariableOp_7вAssignVariableOp_70вAssignVariableOp_71вAssignVariableOp_72вAssignVariableOp_73вAssignVariableOp_74вAssignVariableOp_75вAssignVariableOp_76вAssignVariableOp_77вAssignVariableOp_78вAssignVariableOp_79вAssignVariableOp_8вAssignVariableOp_80вAssignVariableOp_9в	RestoreV2вRestoreV2_1╩-
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:Q*
dtype0*╓,
value╠,B╔,QB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names│
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:Q*
dtype0*╖
valueнBкQB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices├
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*┌
_output_shapes╟
─:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*_
dtypesU
S2Q	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

IdentityР
AssignVariableOpAssignVariableOp assignvariableop_conv3d_4_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1Ц
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv3d_4_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2д
AssignVariableOp_2AssignVariableOp.assignvariableop_2_batch_normalization_4_gammaIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3г
AssignVariableOp_3AssignVariableOp-assignvariableop_3_batch_normalization_4_betaIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4к
AssignVariableOp_4AssignVariableOp4assignvariableop_4_batch_normalization_4_moving_meanIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5о
AssignVariableOp_5AssignVariableOp8assignvariableop_5_batch_normalization_4_moving_varianceIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6Ш
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv3d_5_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7Ц
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv3d_5_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8д
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_5_gammaIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9г
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_5_betaIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10о
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_5_moving_meanIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11▓
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_5_moving_varianceIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12Ь
AssignVariableOp_12AssignVariableOp#assignvariableop_12_conv3d_6_kernelIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13Ъ
AssignVariableOp_13AssignVariableOp!assignvariableop_13_conv3d_6_biasIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14и
AssignVariableOp_14AssignVariableOp/assignvariableop_14_batch_normalization_6_gammaIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15з
AssignVariableOp_15AssignVariableOp.assignvariableop_15_batch_normalization_6_betaIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16о
AssignVariableOp_16AssignVariableOp5assignvariableop_16_batch_normalization_6_moving_meanIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17▓
AssignVariableOp_17AssignVariableOp9assignvariableop_17_batch_normalization_6_moving_varianceIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18Ь
AssignVariableOp_18AssignVariableOp#assignvariableop_18_conv3d_7_kernelIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19Ъ
AssignVariableOp_19AssignVariableOp!assignvariableop_19_conv3d_7_biasIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20и
AssignVariableOp_20AssignVariableOp/assignvariableop_20_batch_normalization_7_gammaIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21з
AssignVariableOp_21AssignVariableOp.assignvariableop_21_batch_normalization_7_betaIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22о
AssignVariableOp_22AssignVariableOp5assignvariableop_22_batch_normalization_7_moving_meanIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23▓
AssignVariableOp_23AssignVariableOp9assignvariableop_23_batch_normalization_7_moving_varianceIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24Ь
AssignVariableOp_24AssignVariableOp#assignvariableop_24_layer1_1_kernelIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25Ъ
AssignVariableOp_25AssignVariableOp!assignvariableop_25_layer1_1_biasIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26Ь
AssignVariableOp_26AssignVariableOp#assignvariableop_26_layer2_1_kernelIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27Ъ
AssignVariableOp_27AssignVariableOp!assignvariableop_27_layer2_1_biasIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28Ы
AssignVariableOp_28AssignVariableOp"assignvariableop_28_dense_1_kernelIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29Щ
AssignVariableOp_29AssignVariableOp assignvariableop_29_dense_1_biasIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0	*
_output_shapes
:2
Identity_30Ц
AssignVariableOp_30AssignVariableOpassignvariableop_30_adam_iterIdentity_30:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31Ш
AssignVariableOp_31AssignVariableOpassignvariableop_31_adam_beta_1Identity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32Ш
AssignVariableOp_32AssignVariableOpassignvariableop_32_adam_beta_2Identity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33Ч
AssignVariableOp_33AssignVariableOpassignvariableop_33_adam_decayIdentity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34Я
AssignVariableOp_34AssignVariableOp&assignvariableop_34_adam_learning_rateIdentity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35Т
AssignVariableOp_35AssignVariableOpassignvariableop_35_totalIdentity_35:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_35_
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:2
Identity_36Т
AssignVariableOp_36AssignVariableOpassignvariableop_36_countIdentity_36:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_36_
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:2
Identity_37г
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_conv3d_4_kernel_mIdentity_37:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_37_
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:2
Identity_38б
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_conv3d_4_bias_mIdentity_38:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_38_
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:2
Identity_39п
AssignVariableOp_39AssignVariableOp6assignvariableop_39_adam_batch_normalization_4_gamma_mIdentity_39:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_39_
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:2
Identity_40о
AssignVariableOp_40AssignVariableOp5assignvariableop_40_adam_batch_normalization_4_beta_mIdentity_40:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_40_
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:2
Identity_41г
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_conv3d_5_kernel_mIdentity_41:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_41_
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:2
Identity_42б
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_conv3d_5_bias_mIdentity_42:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_42_
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:2
Identity_43п
AssignVariableOp_43AssignVariableOp6assignvariableop_43_adam_batch_normalization_5_gamma_mIdentity_43:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_43_
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:2
Identity_44о
AssignVariableOp_44AssignVariableOp5assignvariableop_44_adam_batch_normalization_5_beta_mIdentity_44:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_44_
Identity_45IdentityRestoreV2:tensors:45*
T0*
_output_shapes
:2
Identity_45г
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_conv3d_6_kernel_mIdentity_45:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_45_
Identity_46IdentityRestoreV2:tensors:46*
T0*
_output_shapes
:2
Identity_46б
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_conv3d_6_bias_mIdentity_46:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_46_
Identity_47IdentityRestoreV2:tensors:47*
T0*
_output_shapes
:2
Identity_47п
AssignVariableOp_47AssignVariableOp6assignvariableop_47_adam_batch_normalization_6_gamma_mIdentity_47:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_47_
Identity_48IdentityRestoreV2:tensors:48*
T0*
_output_shapes
:2
Identity_48о
AssignVariableOp_48AssignVariableOp5assignvariableop_48_adam_batch_normalization_6_beta_mIdentity_48:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_48_
Identity_49IdentityRestoreV2:tensors:49*
T0*
_output_shapes
:2
Identity_49г
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_conv3d_7_kernel_mIdentity_49:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_49_
Identity_50IdentityRestoreV2:tensors:50*
T0*
_output_shapes
:2
Identity_50б
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_conv3d_7_bias_mIdentity_50:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_50_
Identity_51IdentityRestoreV2:tensors:51*
T0*
_output_shapes
:2
Identity_51п
AssignVariableOp_51AssignVariableOp6assignvariableop_51_adam_batch_normalization_7_gamma_mIdentity_51:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_51_
Identity_52IdentityRestoreV2:tensors:52*
T0*
_output_shapes
:2
Identity_52о
AssignVariableOp_52AssignVariableOp5assignvariableop_52_adam_batch_normalization_7_beta_mIdentity_52:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_52_
Identity_53IdentityRestoreV2:tensors:53*
T0*
_output_shapes
:2
Identity_53г
AssignVariableOp_53AssignVariableOp*assignvariableop_53_adam_layer1_1_kernel_mIdentity_53:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_53_
Identity_54IdentityRestoreV2:tensors:54*
T0*
_output_shapes
:2
Identity_54б
AssignVariableOp_54AssignVariableOp(assignvariableop_54_adam_layer1_1_bias_mIdentity_54:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_54_
Identity_55IdentityRestoreV2:tensors:55*
T0*
_output_shapes
:2
Identity_55г
AssignVariableOp_55AssignVariableOp*assignvariableop_55_adam_layer2_1_kernel_mIdentity_55:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_55_
Identity_56IdentityRestoreV2:tensors:56*
T0*
_output_shapes
:2
Identity_56б
AssignVariableOp_56AssignVariableOp(assignvariableop_56_adam_layer2_1_bias_mIdentity_56:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_56_
Identity_57IdentityRestoreV2:tensors:57*
T0*
_output_shapes
:2
Identity_57в
AssignVariableOp_57AssignVariableOp)assignvariableop_57_adam_dense_1_kernel_mIdentity_57:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_57_
Identity_58IdentityRestoreV2:tensors:58*
T0*
_output_shapes
:2
Identity_58а
AssignVariableOp_58AssignVariableOp'assignvariableop_58_adam_dense_1_bias_mIdentity_58:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_58_
Identity_59IdentityRestoreV2:tensors:59*
T0*
_output_shapes
:2
Identity_59г
AssignVariableOp_59AssignVariableOp*assignvariableop_59_adam_conv3d_4_kernel_vIdentity_59:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_59_
Identity_60IdentityRestoreV2:tensors:60*
T0*
_output_shapes
:2
Identity_60б
AssignVariableOp_60AssignVariableOp(assignvariableop_60_adam_conv3d_4_bias_vIdentity_60:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_60_
Identity_61IdentityRestoreV2:tensors:61*
T0*
_output_shapes
:2
Identity_61п
AssignVariableOp_61AssignVariableOp6assignvariableop_61_adam_batch_normalization_4_gamma_vIdentity_61:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_61_
Identity_62IdentityRestoreV2:tensors:62*
T0*
_output_shapes
:2
Identity_62о
AssignVariableOp_62AssignVariableOp5assignvariableop_62_adam_batch_normalization_4_beta_vIdentity_62:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_62_
Identity_63IdentityRestoreV2:tensors:63*
T0*
_output_shapes
:2
Identity_63г
AssignVariableOp_63AssignVariableOp*assignvariableop_63_adam_conv3d_5_kernel_vIdentity_63:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_63_
Identity_64IdentityRestoreV2:tensors:64*
T0*
_output_shapes
:2
Identity_64б
AssignVariableOp_64AssignVariableOp(assignvariableop_64_adam_conv3d_5_bias_vIdentity_64:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_64_
Identity_65IdentityRestoreV2:tensors:65*
T0*
_output_shapes
:2
Identity_65п
AssignVariableOp_65AssignVariableOp6assignvariableop_65_adam_batch_normalization_5_gamma_vIdentity_65:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_65_
Identity_66IdentityRestoreV2:tensors:66*
T0*
_output_shapes
:2
Identity_66о
AssignVariableOp_66AssignVariableOp5assignvariableop_66_adam_batch_normalization_5_beta_vIdentity_66:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_66_
Identity_67IdentityRestoreV2:tensors:67*
T0*
_output_shapes
:2
Identity_67г
AssignVariableOp_67AssignVariableOp*assignvariableop_67_adam_conv3d_6_kernel_vIdentity_67:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_67_
Identity_68IdentityRestoreV2:tensors:68*
T0*
_output_shapes
:2
Identity_68б
AssignVariableOp_68AssignVariableOp(assignvariableop_68_adam_conv3d_6_bias_vIdentity_68:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_68_
Identity_69IdentityRestoreV2:tensors:69*
T0*
_output_shapes
:2
Identity_69п
AssignVariableOp_69AssignVariableOp6assignvariableop_69_adam_batch_normalization_6_gamma_vIdentity_69:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_69_
Identity_70IdentityRestoreV2:tensors:70*
T0*
_output_shapes
:2
Identity_70о
AssignVariableOp_70AssignVariableOp5assignvariableop_70_adam_batch_normalization_6_beta_vIdentity_70:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_70_
Identity_71IdentityRestoreV2:tensors:71*
T0*
_output_shapes
:2
Identity_71г
AssignVariableOp_71AssignVariableOp*assignvariableop_71_adam_conv3d_7_kernel_vIdentity_71:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_71_
Identity_72IdentityRestoreV2:tensors:72*
T0*
_output_shapes
:2
Identity_72б
AssignVariableOp_72AssignVariableOp(assignvariableop_72_adam_conv3d_7_bias_vIdentity_72:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_72_
Identity_73IdentityRestoreV2:tensors:73*
T0*
_output_shapes
:2
Identity_73п
AssignVariableOp_73AssignVariableOp6assignvariableop_73_adam_batch_normalization_7_gamma_vIdentity_73:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_73_
Identity_74IdentityRestoreV2:tensors:74*
T0*
_output_shapes
:2
Identity_74о
AssignVariableOp_74AssignVariableOp5assignvariableop_74_adam_batch_normalization_7_beta_vIdentity_74:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_74_
Identity_75IdentityRestoreV2:tensors:75*
T0*
_output_shapes
:2
Identity_75г
AssignVariableOp_75AssignVariableOp*assignvariableop_75_adam_layer1_1_kernel_vIdentity_75:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_75_
Identity_76IdentityRestoreV2:tensors:76*
T0*
_output_shapes
:2
Identity_76б
AssignVariableOp_76AssignVariableOp(assignvariableop_76_adam_layer1_1_bias_vIdentity_76:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_76_
Identity_77IdentityRestoreV2:tensors:77*
T0*
_output_shapes
:2
Identity_77г
AssignVariableOp_77AssignVariableOp*assignvariableop_77_adam_layer2_1_kernel_vIdentity_77:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_77_
Identity_78IdentityRestoreV2:tensors:78*
T0*
_output_shapes
:2
Identity_78б
AssignVariableOp_78AssignVariableOp(assignvariableop_78_adam_layer2_1_bias_vIdentity_78:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_78_
Identity_79IdentityRestoreV2:tensors:79*
T0*
_output_shapes
:2
Identity_79в
AssignVariableOp_79AssignVariableOp)assignvariableop_79_adam_dense_1_kernel_vIdentity_79:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_79_
Identity_80IdentityRestoreV2:tensors:80*
T0*
_output_shapes
:2
Identity_80а
AssignVariableOp_80AssignVariableOp'assignvariableop_80_adam_dense_1_bias_vIdentity_80:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_80и
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_namesФ
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices─
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp╘
Identity_81Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_81с
Identity_82IdentityIdentity_81:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_82"#
identity_82Identity_82:output:0*█
_input_shapes╔
╞: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: :1

_output_shapes
: :2

_output_shapes
: :3

_output_shapes
: :4

_output_shapes
: :5

_output_shapes
: :6

_output_shapes
: :7

_output_shapes
: :8

_output_shapes
: :9

_output_shapes
: ::

_output_shapes
: :;

_output_shapes
: :<

_output_shapes
: :=

_output_shapes
: :>

_output_shapes
: :?

_output_shapes
: :@

_output_shapes
: :A

_output_shapes
: :B

_output_shapes
: :C

_output_shapes
: :D

_output_shapes
: :E

_output_shapes
: :F

_output_shapes
: :G

_output_shapes
: :H

_output_shapes
: :I

_output_shapes
: :J

_output_shapes
: :K

_output_shapes
: :L

_output_shapes
: :M

_output_shapes
: :N

_output_shapes
: :O

_output_shapes
: :P

_output_shapes
: :Q

_output_shapes
: 
ш
н
E__inference_dense_1_layer_call_and_return_conditional_losses_23020217

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         :::O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
╬
e
G__inference_dropout_3_layer_call_and_return_conditional_losses_23022128

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         А
2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А
2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:         А
:P L
(
_output_shapes
:         А

 
_user_specified_nameinputs
М
f
G__inference_dropout_3_layer_call_and_return_conditional_losses_23022123

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU╒?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         А
2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╡
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А
*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠>2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А
2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А
2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         А
2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         А
2

Identity"
identityIdentity:output:0*'
_input_shapes
:         А
:P L
(
_output_shapes
:         А

 
_user_specified_nameinputs
АZ
Ъ
E__inference_model_1_layer_call_and_return_conditional_losses_23020316
input_3
input_4
conv3d_4_23020238
conv3d_4_23020240"
batch_normalization_4_23020243"
batch_normalization_4_23020245"
batch_normalization_4_23020247"
batch_normalization_4_23020249
conv3d_5_23020252
conv3d_5_23020254"
batch_normalization_5_23020257"
batch_normalization_5_23020259"
batch_normalization_5_23020261"
batch_normalization_5_23020263
conv3d_6_23020266
conv3d_6_23020268"
batch_normalization_6_23020271"
batch_normalization_6_23020273"
batch_normalization_6_23020275"
batch_normalization_6_23020277
conv3d_7_23020280
conv3d_7_23020282"
batch_normalization_7_23020285"
batch_normalization_7_23020287"
batch_normalization_7_23020289"
batch_normalization_7_23020291
layer1_23020298
layer1_23020300
layer2_23020304
layer2_23020306
dense_1_23020310
dense_1_23020312
identityИв-batch_normalization_4/StatefulPartitionedCallв-batch_normalization_5/StatefulPartitionedCallв-batch_normalization_6/StatefulPartitionedCallв-batch_normalization_7/StatefulPartitionedCallв conv3d_4/StatefulPartitionedCallв conv3d_5/StatefulPartitionedCallв conv3d_6/StatefulPartitionedCallв conv3d_7/StatefulPartitionedCallвdense_1/StatefulPartitionedCallвlayer1/StatefulPartitionedCallвlayer2/StatefulPartitionedCallИ
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCallinput_3conv3d_4_23020238conv3d_4_23020240*
Tin
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_conv3d_4_layer_call_and_return_conditional_losses_230189942"
 conv3d_4/StatefulPartitionedCallп
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0batch_normalization_4_23020243batch_normalization_4_23020245batch_normalization_4_23020247batch_normalization_4_23020249*
Tin	
2*
Tout
2*3
_output_shapes!
:         *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_230197072/
-batch_normalization_4/StatefulPartitionedCall╖
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0conv3d_5_23020252conv3d_5_23020254*
Tin
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_conv3d_5_layer_call_and_return_conditional_losses_230191562"
 conv3d_5/StatefulPartitionedCallп
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv3d_5/StatefulPartitionedCall:output:0batch_normalization_5_23020257batch_normalization_5_23020259batch_normalization_5_23020261batch_normalization_5_23020263*
Tin	
2*
Tout
2*3
_output_shapes!
:         *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_230198032/
-batch_normalization_5/StatefulPartitionedCall╖
 conv3d_6/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0conv3d_6_23020266conv3d_6_23020268*
Tin
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_conv3d_6_layer_call_and_return_conditional_losses_230193182"
 conv3d_6/StatefulPartitionedCallп
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv3d_6/StatefulPartitionedCall:output:0batch_normalization_6_23020271batch_normalization_6_23020273batch_normalization_6_23020275batch_normalization_6_23020277*
Tin	
2*
Tout
2*3
_output_shapes!
:         *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_230198992/
-batch_normalization_6/StatefulPartitionedCall╖
 conv3d_7/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0conv3d_7_23020280conv3d_7_23020282*
Tin
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_conv3d_7_layer_call_and_return_conditional_losses_230194802"
 conv3d_7/StatefulPartitionedCallп
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)conv3d_7/StatefulPartitionedCall:output:0batch_normalization_7_23020285batch_normalization_7_23020287batch_normalization_7_23020289batch_normalization_7_23020291*
Tin	
2*
Tout
2*3
_output_shapes!
:         *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_230199952/
-batch_normalization_7/StatefulPartitionedCallФ
#average_pooling3d_1/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_average_pooling3d_1_layer_call_and_return_conditional_losses_230196362%
#average_pooling3d_1/PartitionedCallс
flatten_1/PartitionedCallPartitionedCall,average_pooling3d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:         А
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_flatten_1_layer_call_and_return_conditional_losses_230200382
flatten_1/PartitionedCall╫
dropout_3/PartitionedCallPartitionedCall"flatten_1/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:         А
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_dropout_3_layer_call_and_return_conditional_losses_230200632
dropout_3/PartitionedCallэ
concatenate_1/PartitionedCallPartitionedCall"dropout_3/PartitionedCall:output:0input_4*
Tin
2*
Tout
2*(
_output_shapes
:         Г
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_concatenate_1_layer_call_and_return_conditional_losses_230200832
concatenate_1/PartitionedCallТ
layer1/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0layer1_23020298layer1_23020300*
Tin
2*
Tout
2*(
_output_shapes
:         ╚*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_layer1_layer_call_and_return_conditional_losses_230201032 
layer1/StatefulPartitionedCall▄
dropout_4/PartitionedCallPartitionedCall'layer1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:         ╚* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_dropout_4_layer_call_and_return_conditional_losses_230201362
dropout_4/PartitionedCallН
layer2/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0layer2_23020304layer2_23020306*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_layer2_layer_call_and_return_conditional_losses_230201602 
layer2/StatefulPartitionedCall█
dropout_5/PartitionedCallPartitionedCall'layer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_dropout_5_layer_call_and_return_conditional_losses_230201932
dropout_5/PartitionedCallТ
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_1_23020310dense_1_23020312*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_230202172!
dense_1/StatefulPartitionedCallм
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall!^conv3d_6/StatefulPartitionedCall!^conv3d_7/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^layer1/StatefulPartitionedCall^layer2/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*└
_input_shapesо
л:         н:         ::::::::::::::::::::::::::::::2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall2D
 conv3d_6/StatefulPartitionedCall conv3d_6/StatefulPartitionedCall2D
 conv3d_7/StatefulPartitionedCall conv3d_7/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall:] Y
4
_output_shapes"
 :         н
!
_user_specified_name	input_3:PL
'
_output_shapes
:         
!
_user_specified_name	input_4:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
о	
л
8__inference_batch_normalization_5_layer_call_fn_23021605

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*N
_output_shapes<
::8                                    *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_230192622
StatefulPartitionedCall╡
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8                                    ::::22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
щ,
╨
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_23019424

inputs
assignmovingavg_23019399
assignmovingavg_1_23019405)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpЩ
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indicesЫ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/meanИ
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient╦
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*N
_output_shapes<
::8                                    2
moments/SquaredDifferenceб
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices╛
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/varianceГ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЛ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1а
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/23019399*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayХ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_23019399*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp┼
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/23019399*
_output_shapes
:2
AssignMovingAvg/sub╝
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/23019399*
_output_shapes
:2
AssignMovingAvg/mulЕ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_23019399AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/23019399*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpж
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/23019405*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЫ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_23019405*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╧
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/23019405*
_output_shapes
:2
AssignMovingAvg_1/sub╞
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/23019405*
_output_shapes
:2
AssignMovingAvg_1/mulС
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_23019405AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/23019405*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulЭ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subм
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/add_1▄
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8                                    ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
╟+
╨
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_23021454

inputs
assignmovingavg_23021429
assignmovingavg_1_23021435)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpЩ
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indicesЫ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/meanИ
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient░
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*3
_output_shapes!
:         2
moments/SquaredDifferenceб
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices╛
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/varianceГ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЛ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1а
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/23021429*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayХ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_23021429*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp┼
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/23021429*
_output_shapes
:2
AssignMovingAvg/sub╝
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/23021429*
_output_shapes
:2
AssignMovingAvg/mulЕ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_23021429AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/23021429*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpж
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/23021435*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЫ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_23021435*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╧
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/23021435*
_output_shapes
:2
AssignMovingAvg_1/sub╞
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/23021435*
_output_shapes
:2
AssignMovingAvg_1/mulС
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_23021435AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/23021435*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulВ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:         2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subС
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2
batchnorm/add_1┴
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:         ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
║
Ц
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_23022074

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulВ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:         2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subС
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:         :::::[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
╟+
╨
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_23022054

inputs
assignmovingavg_23022029
assignmovingavg_1_23022035)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpЩ
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indicesЫ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/meanИ
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient░
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*3
_output_shapes!
:         2
moments/SquaredDifferenceб
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices╛
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/varianceГ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЛ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1а
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/23022029*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayХ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_23022029*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp┼
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/23022029*
_output_shapes
:2
AssignMovingAvg/sub╝
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/23022029*
_output_shapes
:2
AssignMovingAvg/mulЕ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_23022029AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/23022029*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpж
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/23022035*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЫ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_23022035*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╧
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/23022035*
_output_shapes
:2
AssignMovingAvg_1/sub╞
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/23022035*
_output_shapes
:2
AssignMovingAvg_1/mulС
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_23022035AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/23022035*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulВ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:         2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subС
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2
batchnorm/add_1┴
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:         ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
─
л
8__inference_batch_normalization_5_layer_call_fn_23021700

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallК
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*3
_output_shapes!
:         *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_230198032
StatefulPartitionedCallЪ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:         ::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
о	
л
8__inference_batch_normalization_6_layer_call_fn_23021805

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*N
_output_shapes<
::8                                    *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_230194242
StatefulPartitionedCall╡
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8                                    ::::22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
∙
H
,__inference_dropout_5_layer_call_fn_23022245

inputs
identityж
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_dropout_5_layer_call_and_return_conditional_losses_230201932
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╟+
╨
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_23019783

inputs
assignmovingavg_23019758
assignmovingavg_1_23019764)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpЩ
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indicesЫ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/meanИ
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient░
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*3
_output_shapes!
:         2
moments/SquaredDifferenceб
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices╛
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/varianceГ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЛ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1а
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/23019758*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayХ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_23019758*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp┼
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/23019758*
_output_shapes
:2
AssignMovingAvg/sub╝
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/23019758*
_output_shapes
:2
AssignMovingAvg/mulЕ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_23019758AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/23019758*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpж
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/23019764*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЫ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_23019764*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╧
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/23019764*
_output_shapes
:2
AssignMovingAvg_1/sub╞
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/23019764*
_output_shapes
:2
AssignMovingAvg_1/mulС
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_23019764AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/23019764*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulВ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:         2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subС
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2
batchnorm/add_1┴
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:         ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ъ
А
+__inference_conv3d_5_layer_call_fn_23019166

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall■
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*N
_output_shapes<
::8                                    *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_conv3d_5_layer_call_and_return_conditional_losses_230191562
StatefulPartitionedCall╡
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8                                    ::22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
║
Ц
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_23021674

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulВ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:         2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subС
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:         :::::[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Е
e
,__inference_dropout_5_layer_call_fn_23022240

inputs
identityИвStatefulPartitionedCall╛
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_dropout_5_layer_call_and_return_conditional_losses_230201882
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ш
н
E__inference_dense_1_layer_call_and_return_conditional_losses_23022256

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         :::O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
╬
R
6__inference_average_pooling3d_1_layer_call_fn_23019642

inputs
identityр
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*W
_output_shapesE
C:A                                             * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_average_pooling3d_1_layer_call_and_return_conditional_losses_230196362
PartitionedCallЬ
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A                                             2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:A                                             : {
W
_output_shapesE
C:A                                             
 
_user_specified_nameinputs
Й
e
,__inference_dropout_4_layer_call_fn_23022193

inputs
identityИвStatefulPartitionedCall┐
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:         ╚* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_dropout_4_layer_call_and_return_conditional_losses_230201312
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ╚2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ╚22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs
┼│
г
#__inference__wrapped_model_23018983
input_3
input_43
/model_1_conv3d_4_conv3d_readvariableop_resource4
0model_1_conv3d_4_biasadd_readvariableop_resourceC
?model_1_batch_normalization_4_batchnorm_readvariableop_resourceG
Cmodel_1_batch_normalization_4_batchnorm_mul_readvariableop_resourceE
Amodel_1_batch_normalization_4_batchnorm_readvariableop_1_resourceE
Amodel_1_batch_normalization_4_batchnorm_readvariableop_2_resource3
/model_1_conv3d_5_conv3d_readvariableop_resource4
0model_1_conv3d_5_biasadd_readvariableop_resourceC
?model_1_batch_normalization_5_batchnorm_readvariableop_resourceG
Cmodel_1_batch_normalization_5_batchnorm_mul_readvariableop_resourceE
Amodel_1_batch_normalization_5_batchnorm_readvariableop_1_resourceE
Amodel_1_batch_normalization_5_batchnorm_readvariableop_2_resource3
/model_1_conv3d_6_conv3d_readvariableop_resource4
0model_1_conv3d_6_biasadd_readvariableop_resourceC
?model_1_batch_normalization_6_batchnorm_readvariableop_resourceG
Cmodel_1_batch_normalization_6_batchnorm_mul_readvariableop_resourceE
Amodel_1_batch_normalization_6_batchnorm_readvariableop_1_resourceE
Amodel_1_batch_normalization_6_batchnorm_readvariableop_2_resource3
/model_1_conv3d_7_conv3d_readvariableop_resource4
0model_1_conv3d_7_biasadd_readvariableop_resourceC
?model_1_batch_normalization_7_batchnorm_readvariableop_resourceG
Cmodel_1_batch_normalization_7_batchnorm_mul_readvariableop_resourceE
Amodel_1_batch_normalization_7_batchnorm_readvariableop_1_resourceE
Amodel_1_batch_normalization_7_batchnorm_readvariableop_2_resource1
-model_1_layer1_matmul_readvariableop_resource2
.model_1_layer1_biasadd_readvariableop_resource1
-model_1_layer2_matmul_readvariableop_resource2
.model_1_layer2_biasadd_readvariableop_resource2
.model_1_dense_1_matmul_readvariableop_resource3
/model_1_dense_1_biasadd_readvariableop_resource
identityИ═
&model_1/conv3d_4/Conv3D/ReadVariableOpReadVariableOp/model_1_conv3d_4_conv3d_readvariableop_resource*+
_output_shapes
:н*
dtype02(
&model_1/conv3d_4/Conv3D/ReadVariableOp▌
model_1/conv3d_4/Conv3DConv3Dinput_3.model_1/conv3d_4/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         *
paddingVALID*
strides	
2
model_1/conv3d_4/Conv3D┐
'model_1/conv3d_4/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv3d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_1/conv3d_4/BiasAdd/ReadVariableOp╨
model_1/conv3d_4/BiasAddBiasAdd model_1/conv3d_4/Conv3D:output:0/model_1/conv3d_4/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         2
model_1/conv3d_4/BiasAddь
6model_1/batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp?model_1_batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype028
6model_1/batch_normalization_4/batchnorm/ReadVariableOpг
-model_1/batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2/
-model_1/batch_normalization_4/batchnorm/add/yА
+model_1/batch_normalization_4/batchnorm/addAddV2>model_1/batch_normalization_4/batchnorm/ReadVariableOp:value:06model_1/batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
:2-
+model_1/batch_normalization_4/batchnorm/add╜
-model_1/batch_normalization_4/batchnorm/RsqrtRsqrt/model_1/batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
:2/
-model_1/batch_normalization_4/batchnorm/Rsqrt°
:model_1/batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_1_batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02<
:model_1/batch_normalization_4/batchnorm/mul/ReadVariableOp¤
+model_1/batch_normalization_4/batchnorm/mulMul1model_1/batch_normalization_4/batchnorm/Rsqrt:y:0Bmodel_1/batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2-
+model_1/batch_normalization_4/batchnorm/mulў
-model_1/batch_normalization_4/batchnorm/mul_1Mul!model_1/conv3d_4/BiasAdd:output:0/model_1/batch_normalization_4/batchnorm/mul:z:0*
T0*3
_output_shapes!
:         2/
-model_1/batch_normalization_4/batchnorm/mul_1Є
8model_1/batch_normalization_4/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_1_batch_normalization_4_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8model_1/batch_normalization_4/batchnorm/ReadVariableOp_1¤
-model_1/batch_normalization_4/batchnorm/mul_2Mul@model_1/batch_normalization_4/batchnorm/ReadVariableOp_1:value:0/model_1/batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
:2/
-model_1/batch_normalization_4/batchnorm/mul_2Є
8model_1/batch_normalization_4/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_1_batch_normalization_4_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02:
8model_1/batch_normalization_4/batchnorm/ReadVariableOp_2√
+model_1/batch_normalization_4/batchnorm/subSub@model_1/batch_normalization_4/batchnorm/ReadVariableOp_2:value:01model_1/batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2-
+model_1/batch_normalization_4/batchnorm/subЙ
-model_1/batch_normalization_4/batchnorm/add_1AddV21model_1/batch_normalization_4/batchnorm/mul_1:z:0/model_1/batch_normalization_4/batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2/
-model_1/batch_normalization_4/batchnorm/add_1╠
&model_1/conv3d_5/Conv3D/ReadVariableOpReadVariableOp/model_1_conv3d_5_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02(
&model_1/conv3d_5/Conv3D/ReadVariableOpЗ
model_1/conv3d_5/Conv3DConv3D1model_1/batch_normalization_4/batchnorm/add_1:z:0.model_1/conv3d_5/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         *
paddingVALID*
strides	
2
model_1/conv3d_5/Conv3D┐
'model_1/conv3d_5/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv3d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_1/conv3d_5/BiasAdd/ReadVariableOp╨
model_1/conv3d_5/BiasAddBiasAdd model_1/conv3d_5/Conv3D:output:0/model_1/conv3d_5/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         2
model_1/conv3d_5/BiasAddФ
model_1/conv3d_5/EluElu!model_1/conv3d_5/BiasAdd:output:0*
T0*3
_output_shapes!
:         2
model_1/conv3d_5/Eluь
6model_1/batch_normalization_5/batchnorm/ReadVariableOpReadVariableOp?model_1_batch_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype028
6model_1/batch_normalization_5/batchnorm/ReadVariableOpг
-model_1/batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2/
-model_1/batch_normalization_5/batchnorm/add/yА
+model_1/batch_normalization_5/batchnorm/addAddV2>model_1/batch_normalization_5/batchnorm/ReadVariableOp:value:06model_1/batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes
:2-
+model_1/batch_normalization_5/batchnorm/add╜
-model_1/batch_normalization_5/batchnorm/RsqrtRsqrt/model_1/batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
:2/
-model_1/batch_normalization_5/batchnorm/Rsqrt°
:model_1/batch_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_1_batch_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02<
:model_1/batch_normalization_5/batchnorm/mul/ReadVariableOp¤
+model_1/batch_normalization_5/batchnorm/mulMul1model_1/batch_normalization_5/batchnorm/Rsqrt:y:0Bmodel_1/batch_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2-
+model_1/batch_normalization_5/batchnorm/mul°
-model_1/batch_normalization_5/batchnorm/mul_1Mul"model_1/conv3d_5/Elu:activations:0/model_1/batch_normalization_5/batchnorm/mul:z:0*
T0*3
_output_shapes!
:         2/
-model_1/batch_normalization_5/batchnorm/mul_1Є
8model_1/batch_normalization_5/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_1_batch_normalization_5_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8model_1/batch_normalization_5/batchnorm/ReadVariableOp_1¤
-model_1/batch_normalization_5/batchnorm/mul_2Mul@model_1/batch_normalization_5/batchnorm/ReadVariableOp_1:value:0/model_1/batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
:2/
-model_1/batch_normalization_5/batchnorm/mul_2Є
8model_1/batch_normalization_5/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_1_batch_normalization_5_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02:
8model_1/batch_normalization_5/batchnorm/ReadVariableOp_2√
+model_1/batch_normalization_5/batchnorm/subSub@model_1/batch_normalization_5/batchnorm/ReadVariableOp_2:value:01model_1/batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2-
+model_1/batch_normalization_5/batchnorm/subЙ
-model_1/batch_normalization_5/batchnorm/add_1AddV21model_1/batch_normalization_5/batchnorm/mul_1:z:0/model_1/batch_normalization_5/batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2/
-model_1/batch_normalization_5/batchnorm/add_1╠
&model_1/conv3d_6/Conv3D/ReadVariableOpReadVariableOp/model_1_conv3d_6_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02(
&model_1/conv3d_6/Conv3D/ReadVariableOpЗ
model_1/conv3d_6/Conv3DConv3D1model_1/batch_normalization_5/batchnorm/add_1:z:0.model_1/conv3d_6/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         *
paddingVALID*
strides	
2
model_1/conv3d_6/Conv3D┐
'model_1/conv3d_6/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv3d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_1/conv3d_6/BiasAdd/ReadVariableOp╨
model_1/conv3d_6/BiasAddBiasAdd model_1/conv3d_6/Conv3D:output:0/model_1/conv3d_6/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         2
model_1/conv3d_6/BiasAddФ
model_1/conv3d_6/EluElu!model_1/conv3d_6/BiasAdd:output:0*
T0*3
_output_shapes!
:         2
model_1/conv3d_6/Eluь
6model_1/batch_normalization_6/batchnorm/ReadVariableOpReadVariableOp?model_1_batch_normalization_6_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype028
6model_1/batch_normalization_6/batchnorm/ReadVariableOpг
-model_1/batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2/
-model_1/batch_normalization_6/batchnorm/add/yА
+model_1/batch_normalization_6/batchnorm/addAddV2>model_1/batch_normalization_6/batchnorm/ReadVariableOp:value:06model_1/batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes
:2-
+model_1/batch_normalization_6/batchnorm/add╜
-model_1/batch_normalization_6/batchnorm/RsqrtRsqrt/model_1/batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes
:2/
-model_1/batch_normalization_6/batchnorm/Rsqrt°
:model_1/batch_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_1_batch_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02<
:model_1/batch_normalization_6/batchnorm/mul/ReadVariableOp¤
+model_1/batch_normalization_6/batchnorm/mulMul1model_1/batch_normalization_6/batchnorm/Rsqrt:y:0Bmodel_1/batch_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2-
+model_1/batch_normalization_6/batchnorm/mul°
-model_1/batch_normalization_6/batchnorm/mul_1Mul"model_1/conv3d_6/Elu:activations:0/model_1/batch_normalization_6/batchnorm/mul:z:0*
T0*3
_output_shapes!
:         2/
-model_1/batch_normalization_6/batchnorm/mul_1Є
8model_1/batch_normalization_6/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_1_batch_normalization_6_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8model_1/batch_normalization_6/batchnorm/ReadVariableOp_1¤
-model_1/batch_normalization_6/batchnorm/mul_2Mul@model_1/batch_normalization_6/batchnorm/ReadVariableOp_1:value:0/model_1/batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes
:2/
-model_1/batch_normalization_6/batchnorm/mul_2Є
8model_1/batch_normalization_6/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_1_batch_normalization_6_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02:
8model_1/batch_normalization_6/batchnorm/ReadVariableOp_2√
+model_1/batch_normalization_6/batchnorm/subSub@model_1/batch_normalization_6/batchnorm/ReadVariableOp_2:value:01model_1/batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2-
+model_1/batch_normalization_6/batchnorm/subЙ
-model_1/batch_normalization_6/batchnorm/add_1AddV21model_1/batch_normalization_6/batchnorm/mul_1:z:0/model_1/batch_normalization_6/batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2/
-model_1/batch_normalization_6/batchnorm/add_1╠
&model_1/conv3d_7/Conv3D/ReadVariableOpReadVariableOp/model_1_conv3d_7_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02(
&model_1/conv3d_7/Conv3D/ReadVariableOpЗ
model_1/conv3d_7/Conv3DConv3D1model_1/batch_normalization_6/batchnorm/add_1:z:0.model_1/conv3d_7/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         *
paddingVALID*
strides	
2
model_1/conv3d_7/Conv3D┐
'model_1/conv3d_7/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv3d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'model_1/conv3d_7/BiasAdd/ReadVariableOp╨
model_1/conv3d_7/BiasAddBiasAdd model_1/conv3d_7/Conv3D:output:0/model_1/conv3d_7/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         2
model_1/conv3d_7/BiasAddФ
model_1/conv3d_7/EluElu!model_1/conv3d_7/BiasAdd:output:0*
T0*3
_output_shapes!
:         2
model_1/conv3d_7/Eluь
6model_1/batch_normalization_7/batchnorm/ReadVariableOpReadVariableOp?model_1_batch_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype028
6model_1/batch_normalization_7/batchnorm/ReadVariableOpг
-model_1/batch_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2/
-model_1/batch_normalization_7/batchnorm/add/yА
+model_1/batch_normalization_7/batchnorm/addAddV2>model_1/batch_normalization_7/batchnorm/ReadVariableOp:value:06model_1/batch_normalization_7/batchnorm/add/y:output:0*
T0*
_output_shapes
:2-
+model_1/batch_normalization_7/batchnorm/add╜
-model_1/batch_normalization_7/batchnorm/RsqrtRsqrt/model_1/batch_normalization_7/batchnorm/add:z:0*
T0*
_output_shapes
:2/
-model_1/batch_normalization_7/batchnorm/Rsqrt°
:model_1/batch_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_1_batch_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02<
:model_1/batch_normalization_7/batchnorm/mul/ReadVariableOp¤
+model_1/batch_normalization_7/batchnorm/mulMul1model_1/batch_normalization_7/batchnorm/Rsqrt:y:0Bmodel_1/batch_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2-
+model_1/batch_normalization_7/batchnorm/mul°
-model_1/batch_normalization_7/batchnorm/mul_1Mul"model_1/conv3d_7/Elu:activations:0/model_1/batch_normalization_7/batchnorm/mul:z:0*
T0*3
_output_shapes!
:         2/
-model_1/batch_normalization_7/batchnorm/mul_1Є
8model_1/batch_normalization_7/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_1_batch_normalization_7_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8model_1/batch_normalization_7/batchnorm/ReadVariableOp_1¤
-model_1/batch_normalization_7/batchnorm/mul_2Mul@model_1/batch_normalization_7/batchnorm/ReadVariableOp_1:value:0/model_1/batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes
:2/
-model_1/batch_normalization_7/batchnorm/mul_2Є
8model_1/batch_normalization_7/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_1_batch_normalization_7_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02:
8model_1/batch_normalization_7/batchnorm/ReadVariableOp_2√
+model_1/batch_normalization_7/batchnorm/subSub@model_1/batch_normalization_7/batchnorm/ReadVariableOp_2:value:01model_1/batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2-
+model_1/batch_normalization_7/batchnorm/subЙ
-model_1/batch_normalization_7/batchnorm/add_1AddV21model_1/batch_normalization_7/batchnorm/mul_1:z:0/model_1/batch_normalization_7/batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2/
-model_1/batch_normalization_7/batchnorm/add_1К
%model_1/average_pooling3d_1/AvgPool3D	AvgPool3D1model_1/batch_normalization_7/batchnorm/add_1:z:0*
T0*3
_output_shapes!
:         *
ksize	
*
paddingVALID*
strides	
2'
%model_1/average_pooling3d_1/AvgPool3DГ
model_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
model_1/flatten_1/Const╞
model_1/flatten_1/ReshapeReshape.model_1/average_pooling3d_1/AvgPool3D:output:0 model_1/flatten_1/Const:output:0*
T0*(
_output_shapes
:         А
2
model_1/flatten_1/ReshapeЫ
model_1/dropout_3/IdentityIdentity"model_1/flatten_1/Reshape:output:0*
T0*(
_output_shapes
:         А
2
model_1/dropout_3/IdentityИ
!model_1/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_1/concatenate_1/concat/axis▐
model_1/concatenate_1/concatConcatV2#model_1/dropout_3/Identity:output:0input_4*model_1/concatenate_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:         Г
2
model_1/concatenate_1/concat╝
$model_1/layer1/MatMul/ReadVariableOpReadVariableOp-model_1_layer1_matmul_readvariableop_resource* 
_output_shapes
:
Г
╚*
dtype02&
$model_1/layer1/MatMul/ReadVariableOp└
model_1/layer1/MatMulMatMul%model_1/concatenate_1/concat:output:0,model_1/layer1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
model_1/layer1/MatMul║
%model_1/layer1/BiasAdd/ReadVariableOpReadVariableOp.model_1_layer1_biasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype02'
%model_1/layer1/BiasAdd/ReadVariableOp╛
model_1/layer1/BiasAddBiasAddmodel_1/layer1/MatMul:product:0-model_1/layer1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
model_1/layer1/BiasAddГ
model_1/layer1/EluElumodel_1/layer1/BiasAdd:output:0*
T0*(
_output_shapes
:         ╚2
model_1/layer1/EluЩ
model_1/dropout_4/IdentityIdentity model_1/layer1/Elu:activations:0*
T0*(
_output_shapes
:         ╚2
model_1/dropout_4/Identity╗
$model_1/layer2/MatMul/ReadVariableOpReadVariableOp-model_1_layer2_matmul_readvariableop_resource*
_output_shapes
:	╚*
dtype02&
$model_1/layer2/MatMul/ReadVariableOp╜
model_1/layer2/MatMulMatMul#model_1/dropout_4/Identity:output:0,model_1/layer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
model_1/layer2/MatMul╣
%model_1/layer2/BiasAdd/ReadVariableOpReadVariableOp.model_1_layer2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%model_1/layer2/BiasAdd/ReadVariableOp╜
model_1/layer2/BiasAddBiasAddmodel_1/layer2/MatMul:product:0-model_1/layer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
model_1/layer2/BiasAddВ
model_1/layer2/EluElumodel_1/layer2/BiasAdd:output:0*
T0*'
_output_shapes
:         2
model_1/layer2/EluШ
model_1/dropout_5/IdentityIdentity model_1/layer2/Elu:activations:0*
T0*'
_output_shapes
:         2
model_1/dropout_5/Identity╜
%model_1/dense_1/MatMul/ReadVariableOpReadVariableOp.model_1_dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02'
%model_1/dense_1/MatMul/ReadVariableOp└
model_1/dense_1/MatMulMatMul#model_1/dropout_5/Identity:output:0-model_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
model_1/dense_1/MatMul╝
&model_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&model_1/dense_1/BiasAdd/ReadVariableOp┴
model_1/dense_1/BiasAddBiasAdd model_1/dense_1/MatMul:product:0.model_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
model_1/dense_1/BiasAddС
model_1/dense_1/SigmoidSigmoid model_1/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         2
model_1/dense_1/Sigmoido
IdentityIdentitymodel_1/dense_1/Sigmoid:y:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*└
_input_shapesо
л:         н:         :::::::::::::::::::::::::::::::] Y
4
_output_shapes"
 :         н
!
_user_specified_name	input_3:PL
'
_output_shapes
:         
!
_user_specified_name	input_4:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ъ
А
+__inference_conv3d_6_layer_call_fn_23019328

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall■
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*N
_output_shapes<
::8                                    *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_conv3d_6_layer_call_and_return_conditional_losses_230193182
StatefulPartitionedCall╡
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8                                    ::22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Д
о
F__inference_conv3d_6_layer_call_and_return_conditional_losses_23019318

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identityИЩ
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype02
Conv3D/ReadVariableOp─
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8                                    *
paddingVALID*
strides	
2
Conv3DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpз
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8                                    2	
BiasAdd|
EluEluBiasAdd:output:0*
T0*N
_output_shapes<
::8                                    2
EluМ
IdentityIdentityElu:activations:0*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8                                    :::v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
╬
e
G__inference_dropout_4_layer_call_and_return_conditional_losses_23020136

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         ╚2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ╚2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:         ╚:P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs
░	
л
8__inference_batch_normalization_4_layer_call_fn_23021418

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*N
_output_shapes<
::8                                    *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_230191332
StatefulPartitionedCall╡
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8                                    ::::22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
┬
л
8__inference_batch_normalization_5_layer_call_fn_23021687

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallИ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_230197832
StatefulPartitionedCallЪ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:         ::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
щ,
╨
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_23021572

inputs
assignmovingavg_23021547
assignmovingavg_1_23021553)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpЩ
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indicesЫ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/meanИ
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient╦
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*N
_output_shapes<
::8                                    2
moments/SquaredDifferenceб
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices╛
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/varianceГ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЛ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1а
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/23021547*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayХ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_23021547*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp┼
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/23021547*
_output_shapes
:2
AssignMovingAvg/sub╝
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/23021547*
_output_shapes
:2
AssignMovingAvg/mulЕ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_23021547AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/23021547*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpж
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/23021553*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЫ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_23021553*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╧
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/23021553*
_output_shapes
:2
AssignMovingAvg_1/sub╞
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/23021553*
_output_shapes
:2
AssignMovingAvg_1/mulС
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_23021553AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/23021553*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulЭ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subм
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/add_1▄
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8                                    ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
║
Ц
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_23021474

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulВ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:         2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subС
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:         :::::[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
║
Ц
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_23021874

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulВ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:         2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subС
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:         :::::[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
¤
H
,__inference_dropout_3_layer_call_fn_23022138

inputs
identityз
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:         А
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_dropout_3_layer_call_and_return_conditional_losses_230200632
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А
2

Identity"
identityIdentity:output:0*'
_input_shapes
:         А
:P L
(
_output_shapes
:         А

 
_user_specified_nameinputs
Ж
─
*__inference_model_1_layer_call_fn_23020613
input_3
input_4
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28
identityИвStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinput_3input_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28*+
Tin$
"2 *
Tout
2*'
_output_shapes
:         *@
_read_only_resource_inputs"
 	
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_model_1_layer_call_and_return_conditional_losses_230205502
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*└
_input_shapesо
л:         н:         ::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
4
_output_shapes"
 :         н
!
_user_specified_name	input_3:PL
'
_output_shapes
:         
!
_user_specified_name	input_4:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
У
H
,__inference_flatten_1_layer_call_fn_23022111

inputs
identityз
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:         А
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_flatten_1_layer_call_and_return_conditional_losses_230200382
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А
2

Identity"
identityIdentity:output:0*2
_input_shapes!
:         :[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs
╬^
Ж
E__inference_model_1_layer_call_and_return_conditional_losses_23020402

inputs
inputs_1
conv3d_4_23020324
conv3d_4_23020326"
batch_normalization_4_23020329"
batch_normalization_4_23020331"
batch_normalization_4_23020333"
batch_normalization_4_23020335
conv3d_5_23020338
conv3d_5_23020340"
batch_normalization_5_23020343"
batch_normalization_5_23020345"
batch_normalization_5_23020347"
batch_normalization_5_23020349
conv3d_6_23020352
conv3d_6_23020354"
batch_normalization_6_23020357"
batch_normalization_6_23020359"
batch_normalization_6_23020361"
batch_normalization_6_23020363
conv3d_7_23020366
conv3d_7_23020368"
batch_normalization_7_23020371"
batch_normalization_7_23020373"
batch_normalization_7_23020375"
batch_normalization_7_23020377
layer1_23020384
layer1_23020386
layer2_23020390
layer2_23020392
dense_1_23020396
dense_1_23020398
identityИв-batch_normalization_4/StatefulPartitionedCallв-batch_normalization_5/StatefulPartitionedCallв-batch_normalization_6/StatefulPartitionedCallв-batch_normalization_7/StatefulPartitionedCallв conv3d_4/StatefulPartitionedCallв conv3d_5/StatefulPartitionedCallв conv3d_6/StatefulPartitionedCallв conv3d_7/StatefulPartitionedCallвdense_1/StatefulPartitionedCallв!dropout_3/StatefulPartitionedCallв!dropout_4/StatefulPartitionedCallв!dropout_5/StatefulPartitionedCallвlayer1/StatefulPartitionedCallвlayer2/StatefulPartitionedCallЗ
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCallinputsconv3d_4_23020324conv3d_4_23020326*
Tin
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_conv3d_4_layer_call_and_return_conditional_losses_230189942"
 conv3d_4/StatefulPartitionedCallн
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0batch_normalization_4_23020329batch_normalization_4_23020331batch_normalization_4_23020333batch_normalization_4_23020335*
Tin	
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_230196872/
-batch_normalization_4/StatefulPartitionedCall╖
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0conv3d_5_23020338conv3d_5_23020340*
Tin
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_conv3d_5_layer_call_and_return_conditional_losses_230191562"
 conv3d_5/StatefulPartitionedCallн
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv3d_5/StatefulPartitionedCall:output:0batch_normalization_5_23020343batch_normalization_5_23020345batch_normalization_5_23020347batch_normalization_5_23020349*
Tin	
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_230197832/
-batch_normalization_5/StatefulPartitionedCall╖
 conv3d_6/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0conv3d_6_23020352conv3d_6_23020354*
Tin
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_conv3d_6_layer_call_and_return_conditional_losses_230193182"
 conv3d_6/StatefulPartitionedCallн
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv3d_6/StatefulPartitionedCall:output:0batch_normalization_6_23020357batch_normalization_6_23020359batch_normalization_6_23020361batch_normalization_6_23020363*
Tin	
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_230198792/
-batch_normalization_6/StatefulPartitionedCall╖
 conv3d_7/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0conv3d_7_23020366conv3d_7_23020368*
Tin
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_conv3d_7_layer_call_and_return_conditional_losses_230194802"
 conv3d_7/StatefulPartitionedCallн
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)conv3d_7/StatefulPartitionedCall:output:0batch_normalization_7_23020371batch_normalization_7_23020373batch_normalization_7_23020375batch_normalization_7_23020377*
Tin	
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_230199752/
-batch_normalization_7/StatefulPartitionedCallФ
#average_pooling3d_1/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_average_pooling3d_1_layer_call_and_return_conditional_losses_230196362%
#average_pooling3d_1/PartitionedCallс
flatten_1/PartitionedCallPartitionedCall,average_pooling3d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:         А
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_flatten_1_layer_call_and_return_conditional_losses_230200382
flatten_1/PartitionedCallя
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:         А
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_dropout_3_layer_call_and_return_conditional_losses_230200582#
!dropout_3/StatefulPartitionedCallЎ
concatenate_1/PartitionedCallPartitionedCall*dropout_3/StatefulPartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*(
_output_shapes
:         Г
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_concatenate_1_layer_call_and_return_conditional_losses_230200832
concatenate_1/PartitionedCallТ
layer1/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0layer1_23020384layer1_23020386*
Tin
2*
Tout
2*(
_output_shapes
:         ╚*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_layer1_layer_call_and_return_conditional_losses_230201032 
layer1/StatefulPartitionedCallШ
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0"^dropout_3/StatefulPartitionedCall*
Tin
2*
Tout
2*(
_output_shapes
:         ╚* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_dropout_4_layer_call_and_return_conditional_losses_230201312#
!dropout_4/StatefulPartitionedCallХ
layer2/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0layer2_23020390layer2_23020392*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_layer2_layer_call_and_return_conditional_losses_230201602 
layer2/StatefulPartitionedCallЧ
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall'layer2/StatefulPartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_dropout_5_layer_call_and_return_conditional_losses_230201882#
!dropout_5/StatefulPartitionedCallЪ
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0dense_1_23020396dense_1_23020398*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_230202172!
dense_1/StatefulPartitionedCallШ
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall!^conv3d_6/StatefulPartitionedCall!^conv3d_7/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall^layer1/StatefulPartitionedCall^layer2/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*└
_input_shapesо
л:         н:         ::::::::::::::::::::::::::::::2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall2D
 conv3d_6/StatefulPartitionedCall conv3d_6/StatefulPartitionedCall2D
 conv3d_7/StatefulPartitionedCall conv3d_7/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall:\ X
4
_output_shapes"
 :         н
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
К
\
0__inference_concatenate_1_layer_call_fn_23022151
inputs_0
inputs_1
identity╕
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*(
_output_shapes
:         Г
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_concatenate_1_layer_call_and_return_conditional_losses_230200832
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         Г
2

Identity"
identityIdentity:output:0*:
_input_shapes)
':         А
:         :R N
(
_output_shapes
:         А

"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1
╬
e
G__inference_dropout_3_layer_call_and_return_conditional_losses_23020063

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         А
2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А
2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:         А
:P L
(
_output_shapes
:         А

 
_user_specified_nameinputs
┬
Ц
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_23021792

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulЭ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subм
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/add_1О
IdentityIdentitybatchnorm/add_1:z:0*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8                                    :::::v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Д
о
F__inference_conv3d_5_layer_call_and_return_conditional_losses_23019156

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identityИЩ
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource**
_output_shapes
:*
dtype02
Conv3D/ReadVariableOp─
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8                                    *
paddingVALID*
strides	
2
Conv3DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpз
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8                                    2	
BiasAdd|
EluEluBiasAdd:output:0*
T0*N
_output_shapes<
::8                                    2
EluМ
IdentityIdentityElu:activations:0*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8                                    :::v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
■Y
Ъ
E__inference_model_1_layer_call_and_return_conditional_losses_23020550

inputs
inputs_1
conv3d_4_23020472
conv3d_4_23020474"
batch_normalization_4_23020477"
batch_normalization_4_23020479"
batch_normalization_4_23020481"
batch_normalization_4_23020483
conv3d_5_23020486
conv3d_5_23020488"
batch_normalization_5_23020491"
batch_normalization_5_23020493"
batch_normalization_5_23020495"
batch_normalization_5_23020497
conv3d_6_23020500
conv3d_6_23020502"
batch_normalization_6_23020505"
batch_normalization_6_23020507"
batch_normalization_6_23020509"
batch_normalization_6_23020511
conv3d_7_23020514
conv3d_7_23020516"
batch_normalization_7_23020519"
batch_normalization_7_23020521"
batch_normalization_7_23020523"
batch_normalization_7_23020525
layer1_23020532
layer1_23020534
layer2_23020538
layer2_23020540
dense_1_23020544
dense_1_23020546
identityИв-batch_normalization_4/StatefulPartitionedCallв-batch_normalization_5/StatefulPartitionedCallв-batch_normalization_6/StatefulPartitionedCallв-batch_normalization_7/StatefulPartitionedCallв conv3d_4/StatefulPartitionedCallв conv3d_5/StatefulPartitionedCallв conv3d_6/StatefulPartitionedCallв conv3d_7/StatefulPartitionedCallвdense_1/StatefulPartitionedCallвlayer1/StatefulPartitionedCallвlayer2/StatefulPartitionedCallЗ
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCallinputsconv3d_4_23020472conv3d_4_23020474*
Tin
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_conv3d_4_layer_call_and_return_conditional_losses_230189942"
 conv3d_4/StatefulPartitionedCallп
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0batch_normalization_4_23020477batch_normalization_4_23020479batch_normalization_4_23020481batch_normalization_4_23020483*
Tin	
2*
Tout
2*3
_output_shapes!
:         *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_230197072/
-batch_normalization_4/StatefulPartitionedCall╖
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0conv3d_5_23020486conv3d_5_23020488*
Tin
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_conv3d_5_layer_call_and_return_conditional_losses_230191562"
 conv3d_5/StatefulPartitionedCallп
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv3d_5/StatefulPartitionedCall:output:0batch_normalization_5_23020491batch_normalization_5_23020493batch_normalization_5_23020495batch_normalization_5_23020497*
Tin	
2*
Tout
2*3
_output_shapes!
:         *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_230198032/
-batch_normalization_5/StatefulPartitionedCall╖
 conv3d_6/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0conv3d_6_23020500conv3d_6_23020502*
Tin
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_conv3d_6_layer_call_and_return_conditional_losses_230193182"
 conv3d_6/StatefulPartitionedCallп
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv3d_6/StatefulPartitionedCall:output:0batch_normalization_6_23020505batch_normalization_6_23020507batch_normalization_6_23020509batch_normalization_6_23020511*
Tin	
2*
Tout
2*3
_output_shapes!
:         *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_230198992/
-batch_normalization_6/StatefulPartitionedCall╖
 conv3d_7/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0conv3d_7_23020514conv3d_7_23020516*
Tin
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_conv3d_7_layer_call_and_return_conditional_losses_230194802"
 conv3d_7/StatefulPartitionedCallп
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)conv3d_7/StatefulPartitionedCall:output:0batch_normalization_7_23020519batch_normalization_7_23020521batch_normalization_7_23020523batch_normalization_7_23020525*
Tin	
2*
Tout
2*3
_output_shapes!
:         *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_230199952/
-batch_normalization_7/StatefulPartitionedCallФ
#average_pooling3d_1/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_average_pooling3d_1_layer_call_and_return_conditional_losses_230196362%
#average_pooling3d_1/PartitionedCallс
flatten_1/PartitionedCallPartitionedCall,average_pooling3d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:         А
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_flatten_1_layer_call_and_return_conditional_losses_230200382
flatten_1/PartitionedCall╫
dropout_3/PartitionedCallPartitionedCall"flatten_1/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:         А
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_dropout_3_layer_call_and_return_conditional_losses_230200632
dropout_3/PartitionedCallю
concatenate_1/PartitionedCallPartitionedCall"dropout_3/PartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*(
_output_shapes
:         Г
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_concatenate_1_layer_call_and_return_conditional_losses_230200832
concatenate_1/PartitionedCallТ
layer1/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0layer1_23020532layer1_23020534*
Tin
2*
Tout
2*(
_output_shapes
:         ╚*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_layer1_layer_call_and_return_conditional_losses_230201032 
layer1/StatefulPartitionedCall▄
dropout_4/PartitionedCallPartitionedCall'layer1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:         ╚* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_dropout_4_layer_call_and_return_conditional_losses_230201362
dropout_4/PartitionedCallН
layer2/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0layer2_23020538layer2_23020540*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_layer2_layer_call_and_return_conditional_losses_230201602 
layer2/StatefulPartitionedCall█
dropout_5/PartitionedCallPartitionedCall'layer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_dropout_5_layer_call_and_return_conditional_losses_230201932
dropout_5/PartitionedCallТ
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_1_23020544dense_1_23020546*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_230202172!
dense_1/StatefulPartitionedCallм
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall!^conv3d_6/StatefulPartitionedCall!^conv3d_7/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^layer1/StatefulPartitionedCall^layer2/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*└
_input_shapesо
л:         н:         ::::::::::::::::::::::::::::::2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall2D
 conv3d_6/StatefulPartitionedCall conv3d_6/StatefulPartitionedCall2D
 conv3d_7/StatefulPartitionedCall conv3d_7/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall:\ X
4
_output_shapes"
 :         н
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
■
─
*__inference_model_1_layer_call_fn_23020465
input_3
input_4
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28
identityИвStatefulPartitionedCall╪
StatefulPartitionedCallStatefulPartitionedCallinput_3input_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28*+
Tin$
"2 *
Tout
2*'
_output_shapes
:         *8
_read_only_resource_inputs
	*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_model_1_layer_call_and_return_conditional_losses_230204022
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*└
_input_shapesо
л:         н:         ::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
4
_output_shapes"
 :         н
!
_user_specified_name	input_3:PL
'
_output_shapes
:         
!
_user_specified_name	input_4:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
┬
Ц
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_23021392

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulЭ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subм
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/add_1О
IdentityIdentitybatchnorm/add_1:z:0*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8                                    :::::v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
╟+
╨
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_23021854

inputs
assignmovingavg_23021829
assignmovingavg_1_23021835)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpЩ
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indicesЫ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/meanИ
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient░
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*3
_output_shapes!
:         2
moments/SquaredDifferenceб
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices╛
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/varianceГ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЛ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1а
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/23021829*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayХ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_23021829*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp┼
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/23021829*
_output_shapes
:2
AssignMovingAvg/sub╝
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/23021829*
_output_shapes
:2
AssignMovingAvg/mulЕ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_23021829AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/23021829*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpж
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/23021835*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЫ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_23021835*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╧
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/23021835*
_output_shapes
:2
AssignMovingAvg_1/sub╞
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/23021835*
_output_shapes
:2
AssignMovingAvg_1/mulС
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_23021835AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/23021835*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulВ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:         2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subС
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2
batchnorm/add_1┴
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:         ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
╩
e
G__inference_dropout_5_layer_call_and_return_conditional_losses_23022235

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:         2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╟+
╨
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_23019975

inputs
assignmovingavg_23019950
assignmovingavg_1_23019956)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpЩ
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indicesЫ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/meanИ
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient░
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*3
_output_shapes!
:         2
moments/SquaredDifferenceб
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices╛
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/varianceГ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЛ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1а
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/23019950*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayХ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_23019950*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp┼
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/23019950*
_output_shapes
:2
AssignMovingAvg/sub╝
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/23019950*
_output_shapes
:2
AssignMovingAvg/mulЕ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_23019950AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/23019950*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpж
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/23019956*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЫ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_23019956*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╧
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/23019956*
_output_shapes
:2
AssignMovingAvg_1/sub╞
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/23019956*
_output_shapes
:2
AssignMovingAvg_1/mulС
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_23019956AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/23019956*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulВ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:         2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subС
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2
batchnorm/add_1┴
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:         ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
┬
Ц
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_23019295

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulЭ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subм
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/add_1О
IdentityIdentitybatchnorm/add_1:z:0*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8                                    :::::v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
─
w
K__inference_concatenate_1_layer_call_and_return_conditional_losses_23022145
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisВ
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:         Г
2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:         Г
2

Identity"
identityIdentity:output:0*:
_input_shapes)
':         А
:         :R N
(
_output_shapes
:         А

"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1
┬
л
8__inference_batch_normalization_7_layer_call_fn_23022087

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallИ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_230199752
StatefulPartitionedCallЪ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:         ::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
┬
Ц
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_23019133

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulЭ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subм
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/add_1О
IdentityIdentitybatchnorm/add_1:z:0*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8                                    :::::v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
║
Ц
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_23019707

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulВ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:         2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subС
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:         :::::[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
¤
~
)__inference_layer1_layer_call_fn_23022171

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall╓
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:         ╚*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_layer1_layer_call_and_return_conditional_losses_230201032
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ╚2

Identity"
identityIdentity:output:0*/
_input_shapes
:         Г
::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         Г

 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
 ╧
я
E__inference_model_1_layer_call_and_return_conditional_losses_23021043
inputs_0
inputs_1+
'conv3d_4_conv3d_readvariableop_resource,
(conv3d_4_biasadd_readvariableop_resource2
.batch_normalization_4_assignmovingavg_230208514
0batch_normalization_4_assignmovingavg_1_23020857?
;batch_normalization_4_batchnorm_mul_readvariableop_resource;
7batch_normalization_4_batchnorm_readvariableop_resource+
'conv3d_5_conv3d_readvariableop_resource,
(conv3d_5_biasadd_readvariableop_resource2
.batch_normalization_5_assignmovingavg_230208904
0batch_normalization_5_assignmovingavg_1_23020896?
;batch_normalization_5_batchnorm_mul_readvariableop_resource;
7batch_normalization_5_batchnorm_readvariableop_resource+
'conv3d_6_conv3d_readvariableop_resource,
(conv3d_6_biasadd_readvariableop_resource2
.batch_normalization_6_assignmovingavg_230209294
0batch_normalization_6_assignmovingavg_1_23020935?
;batch_normalization_6_batchnorm_mul_readvariableop_resource;
7batch_normalization_6_batchnorm_readvariableop_resource+
'conv3d_7_conv3d_readvariableop_resource,
(conv3d_7_biasadd_readvariableop_resource2
.batch_normalization_7_assignmovingavg_230209684
0batch_normalization_7_assignmovingavg_1_23020974?
;batch_normalization_7_batchnorm_mul_readvariableop_resource;
7batch_normalization_7_batchnorm_readvariableop_resource)
%layer1_matmul_readvariableop_resource*
&layer1_biasadd_readvariableop_resource)
%layer2_matmul_readvariableop_resource*
&layer2_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identityИв9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpв;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpв9batch_normalization_5/AssignMovingAvg/AssignSubVariableOpв;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOpв9batch_normalization_6/AssignMovingAvg/AssignSubVariableOpв;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOpв9batch_normalization_7/AssignMovingAvg/AssignSubVariableOpв;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp╡
conv3d_4/Conv3D/ReadVariableOpReadVariableOp'conv3d_4_conv3d_readvariableop_resource*+
_output_shapes
:н*
dtype02 
conv3d_4/Conv3D/ReadVariableOp╞
conv3d_4/Conv3DConv3Dinputs_0&conv3d_4/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         *
paddingVALID*
strides	
2
conv3d_4/Conv3Dз
conv3d_4/BiasAdd/ReadVariableOpReadVariableOp(conv3d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv3d_4/BiasAdd/ReadVariableOp░
conv3d_4/BiasAddBiasAddconv3d_4/Conv3D:output:0'conv3d_4/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         2
conv3d_4/BiasAdd┼
4batch_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             26
4batch_normalization_4/moments/mean/reduction_indicesЁ
"batch_normalization_4/moments/meanMeanconv3d_4/BiasAdd:output:0=batch_normalization_4/moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2$
"batch_normalization_4/moments/mean╩
*batch_normalization_4/moments/StopGradientStopGradient+batch_normalization_4/moments/mean:output:0*
T0**
_output_shapes
:2,
*batch_normalization_4/moments/StopGradientЕ
/batch_normalization_4/moments/SquaredDifferenceSquaredDifferenceconv3d_4/BiasAdd:output:03batch_normalization_4/moments/StopGradient:output:0*
T0*3
_output_shapes!
:         21
/batch_normalization_4/moments/SquaredDifference═
8batch_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2:
8batch_normalization_4/moments/variance/reduction_indicesЦ
&batch_normalization_4/moments/varianceMean3batch_normalization_4/moments/SquaredDifference:z:0Abatch_normalization_4/moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2(
&batch_normalization_4/moments/variance┼
%batch_normalization_4/moments/SqueezeSqueeze+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2'
%batch_normalization_4/moments/Squeeze═
'batch_normalization_4/moments/Squeeze_1Squeeze/batch_normalization_4/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2)
'batch_normalization_4/moments/Squeeze_1т
+batch_normalization_4/AssignMovingAvg/decayConst*A
_class7
53loc:@batch_normalization_4/AssignMovingAvg/23020851*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2-
+batch_normalization_4/AssignMovingAvg/decay╫
4batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOp.batch_normalization_4_assignmovingavg_23020851*
_output_shapes
:*
dtype026
4batch_normalization_4/AssignMovingAvg/ReadVariableOp│
)batch_normalization_4/AssignMovingAvg/subSub<batch_normalization_4/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_4/moments/Squeeze:output:0*
T0*A
_class7
53loc:@batch_normalization_4/AssignMovingAvg/23020851*
_output_shapes
:2+
)batch_normalization_4/AssignMovingAvg/subк
)batch_normalization_4/AssignMovingAvg/mulMul-batch_normalization_4/AssignMovingAvg/sub:z:04batch_normalization_4/AssignMovingAvg/decay:output:0*
T0*A
_class7
53loc:@batch_normalization_4/AssignMovingAvg/23020851*
_output_shapes
:2+
)batch_normalization_4/AssignMovingAvg/mulЙ
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp.batch_normalization_4_assignmovingavg_23020851-batch_normalization_4/AssignMovingAvg/mul:z:05^batch_normalization_4/AssignMovingAvg/ReadVariableOp*A
_class7
53loc:@batch_normalization_4/AssignMovingAvg/23020851*
_output_shapes
 *
dtype02;
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpш
-batch_normalization_4/AssignMovingAvg_1/decayConst*C
_class9
75loc:@batch_normalization_4/AssignMovingAvg_1/23020857*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2/
-batch_normalization_4/AssignMovingAvg_1/decay▌
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOp0batch_normalization_4_assignmovingavg_1_23020857*
_output_shapes
:*
dtype028
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp╜
+batch_normalization_4/AssignMovingAvg_1/subSub>batch_normalization_4/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_4/moments/Squeeze_1:output:0*
T0*C
_class9
75loc:@batch_normalization_4/AssignMovingAvg_1/23020857*
_output_shapes
:2-
+batch_normalization_4/AssignMovingAvg_1/sub┤
+batch_normalization_4/AssignMovingAvg_1/mulMul/batch_normalization_4/AssignMovingAvg_1/sub:z:06batch_normalization_4/AssignMovingAvg_1/decay:output:0*
T0*C
_class9
75loc:@batch_normalization_4/AssignMovingAvg_1/23020857*
_output_shapes
:2-
+batch_normalization_4/AssignMovingAvg_1/mulХ
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp0batch_normalization_4_assignmovingavg_1_23020857/batch_normalization_4/AssignMovingAvg_1/mul:z:07^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp*C
_class9
75loc:@batch_normalization_4/AssignMovingAvg_1/23020857*
_output_shapes
 *
dtype02=
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpУ
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_4/batchnorm/add/y┌
#batch_normalization_4/batchnorm/addAddV20batch_normalization_4/moments/Squeeze_1:output:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_4/batchnorm/addе
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_4/batchnorm/Rsqrtр
2batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization_4/batchnorm/mul/ReadVariableOp▌
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:0:batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_4/batchnorm/mul╫
%batch_normalization_4/batchnorm/mul_1Mulconv3d_4/BiasAdd:output:0'batch_normalization_4/batchnorm/mul:z:0*
T0*3
_output_shapes!
:         2'
%batch_normalization_4/batchnorm/mul_1╙
%batch_normalization_4/batchnorm/mul_2Mul.batch_normalization_4/moments/Squeeze:output:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_4/batchnorm/mul_2╘
.batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype020
.batch_normalization_4/batchnorm/ReadVariableOp┘
#batch_normalization_4/batchnorm/subSub6batch_normalization_4/batchnorm/ReadVariableOp:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_4/batchnorm/subщ
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2'
%batch_normalization_4/batchnorm/add_1┤
conv3d_5/Conv3D/ReadVariableOpReadVariableOp'conv3d_5_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02 
conv3d_5/Conv3D/ReadVariableOpч
conv3d_5/Conv3DConv3D)batch_normalization_4/batchnorm/add_1:z:0&conv3d_5/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         *
paddingVALID*
strides	
2
conv3d_5/Conv3Dз
conv3d_5/BiasAdd/ReadVariableOpReadVariableOp(conv3d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv3d_5/BiasAdd/ReadVariableOp░
conv3d_5/BiasAddBiasAddconv3d_5/Conv3D:output:0'conv3d_5/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         2
conv3d_5/BiasAdd|
conv3d_5/EluEluconv3d_5/BiasAdd:output:0*
T0*3
_output_shapes!
:         2
conv3d_5/Elu┼
4batch_normalization_5/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             26
4batch_normalization_5/moments/mean/reduction_indicesё
"batch_normalization_5/moments/meanMeanconv3d_5/Elu:activations:0=batch_normalization_5/moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2$
"batch_normalization_5/moments/mean╩
*batch_normalization_5/moments/StopGradientStopGradient+batch_normalization_5/moments/mean:output:0*
T0**
_output_shapes
:2,
*batch_normalization_5/moments/StopGradientЖ
/batch_normalization_5/moments/SquaredDifferenceSquaredDifferenceconv3d_5/Elu:activations:03batch_normalization_5/moments/StopGradient:output:0*
T0*3
_output_shapes!
:         21
/batch_normalization_5/moments/SquaredDifference═
8batch_normalization_5/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2:
8batch_normalization_5/moments/variance/reduction_indicesЦ
&batch_normalization_5/moments/varianceMean3batch_normalization_5/moments/SquaredDifference:z:0Abatch_normalization_5/moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2(
&batch_normalization_5/moments/variance┼
%batch_normalization_5/moments/SqueezeSqueeze+batch_normalization_5/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2'
%batch_normalization_5/moments/Squeeze═
'batch_normalization_5/moments/Squeeze_1Squeeze/batch_normalization_5/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2)
'batch_normalization_5/moments/Squeeze_1т
+batch_normalization_5/AssignMovingAvg/decayConst*A
_class7
53loc:@batch_normalization_5/AssignMovingAvg/23020890*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2-
+batch_normalization_5/AssignMovingAvg/decay╫
4batch_normalization_5/AssignMovingAvg/ReadVariableOpReadVariableOp.batch_normalization_5_assignmovingavg_23020890*
_output_shapes
:*
dtype026
4batch_normalization_5/AssignMovingAvg/ReadVariableOp│
)batch_normalization_5/AssignMovingAvg/subSub<batch_normalization_5/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_5/moments/Squeeze:output:0*
T0*A
_class7
53loc:@batch_normalization_5/AssignMovingAvg/23020890*
_output_shapes
:2+
)batch_normalization_5/AssignMovingAvg/subк
)batch_normalization_5/AssignMovingAvg/mulMul-batch_normalization_5/AssignMovingAvg/sub:z:04batch_normalization_5/AssignMovingAvg/decay:output:0*
T0*A
_class7
53loc:@batch_normalization_5/AssignMovingAvg/23020890*
_output_shapes
:2+
)batch_normalization_5/AssignMovingAvg/mulЙ
9batch_normalization_5/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp.batch_normalization_5_assignmovingavg_23020890-batch_normalization_5/AssignMovingAvg/mul:z:05^batch_normalization_5/AssignMovingAvg/ReadVariableOp*A
_class7
53loc:@batch_normalization_5/AssignMovingAvg/23020890*
_output_shapes
 *
dtype02;
9batch_normalization_5/AssignMovingAvg/AssignSubVariableOpш
-batch_normalization_5/AssignMovingAvg_1/decayConst*C
_class9
75loc:@batch_normalization_5/AssignMovingAvg_1/23020896*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2/
-batch_normalization_5/AssignMovingAvg_1/decay▌
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOpReadVariableOp0batch_normalization_5_assignmovingavg_1_23020896*
_output_shapes
:*
dtype028
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp╜
+batch_normalization_5/AssignMovingAvg_1/subSub>batch_normalization_5/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_5/moments/Squeeze_1:output:0*
T0*C
_class9
75loc:@batch_normalization_5/AssignMovingAvg_1/23020896*
_output_shapes
:2-
+batch_normalization_5/AssignMovingAvg_1/sub┤
+batch_normalization_5/AssignMovingAvg_1/mulMul/batch_normalization_5/AssignMovingAvg_1/sub:z:06batch_normalization_5/AssignMovingAvg_1/decay:output:0*
T0*C
_class9
75loc:@batch_normalization_5/AssignMovingAvg_1/23020896*
_output_shapes
:2-
+batch_normalization_5/AssignMovingAvg_1/mulХ
;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp0batch_normalization_5_assignmovingavg_1_23020896/batch_normalization_5/AssignMovingAvg_1/mul:z:07^batch_normalization_5/AssignMovingAvg_1/ReadVariableOp*C
_class9
75loc:@batch_normalization_5/AssignMovingAvg_1/23020896*
_output_shapes
 *
dtype02=
;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOpУ
%batch_normalization_5/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_5/batchnorm/add/y┌
#batch_normalization_5/batchnorm/addAddV20batch_normalization_5/moments/Squeeze_1:output:0.batch_normalization_5/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_5/batchnorm/addе
%batch_normalization_5/batchnorm/RsqrtRsqrt'batch_normalization_5/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_5/batchnorm/Rsqrtр
2batch_normalization_5/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_5_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization_5/batchnorm/mul/ReadVariableOp▌
#batch_normalization_5/batchnorm/mulMul)batch_normalization_5/batchnorm/Rsqrt:y:0:batch_normalization_5/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_5/batchnorm/mul╪
%batch_normalization_5/batchnorm/mul_1Mulconv3d_5/Elu:activations:0'batch_normalization_5/batchnorm/mul:z:0*
T0*3
_output_shapes!
:         2'
%batch_normalization_5/batchnorm/mul_1╙
%batch_normalization_5/batchnorm/mul_2Mul.batch_normalization_5/moments/Squeeze:output:0'batch_normalization_5/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_5/batchnorm/mul_2╘
.batch_normalization_5/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_5_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype020
.batch_normalization_5/batchnorm/ReadVariableOp┘
#batch_normalization_5/batchnorm/subSub6batch_normalization_5/batchnorm/ReadVariableOp:value:0)batch_normalization_5/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_5/batchnorm/subщ
%batch_normalization_5/batchnorm/add_1AddV2)batch_normalization_5/batchnorm/mul_1:z:0'batch_normalization_5/batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2'
%batch_normalization_5/batchnorm/add_1┤
conv3d_6/Conv3D/ReadVariableOpReadVariableOp'conv3d_6_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02 
conv3d_6/Conv3D/ReadVariableOpч
conv3d_6/Conv3DConv3D)batch_normalization_5/batchnorm/add_1:z:0&conv3d_6/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         *
paddingVALID*
strides	
2
conv3d_6/Conv3Dз
conv3d_6/BiasAdd/ReadVariableOpReadVariableOp(conv3d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv3d_6/BiasAdd/ReadVariableOp░
conv3d_6/BiasAddBiasAddconv3d_6/Conv3D:output:0'conv3d_6/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         2
conv3d_6/BiasAdd|
conv3d_6/EluEluconv3d_6/BiasAdd:output:0*
T0*3
_output_shapes!
:         2
conv3d_6/Elu┼
4batch_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             26
4batch_normalization_6/moments/mean/reduction_indicesё
"batch_normalization_6/moments/meanMeanconv3d_6/Elu:activations:0=batch_normalization_6/moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2$
"batch_normalization_6/moments/mean╩
*batch_normalization_6/moments/StopGradientStopGradient+batch_normalization_6/moments/mean:output:0*
T0**
_output_shapes
:2,
*batch_normalization_6/moments/StopGradientЖ
/batch_normalization_6/moments/SquaredDifferenceSquaredDifferenceconv3d_6/Elu:activations:03batch_normalization_6/moments/StopGradient:output:0*
T0*3
_output_shapes!
:         21
/batch_normalization_6/moments/SquaredDifference═
8batch_normalization_6/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2:
8batch_normalization_6/moments/variance/reduction_indicesЦ
&batch_normalization_6/moments/varianceMean3batch_normalization_6/moments/SquaredDifference:z:0Abatch_normalization_6/moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2(
&batch_normalization_6/moments/variance┼
%batch_normalization_6/moments/SqueezeSqueeze+batch_normalization_6/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2'
%batch_normalization_6/moments/Squeeze═
'batch_normalization_6/moments/Squeeze_1Squeeze/batch_normalization_6/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2)
'batch_normalization_6/moments/Squeeze_1т
+batch_normalization_6/AssignMovingAvg/decayConst*A
_class7
53loc:@batch_normalization_6/AssignMovingAvg/23020929*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2-
+batch_normalization_6/AssignMovingAvg/decay╫
4batch_normalization_6/AssignMovingAvg/ReadVariableOpReadVariableOp.batch_normalization_6_assignmovingavg_23020929*
_output_shapes
:*
dtype026
4batch_normalization_6/AssignMovingAvg/ReadVariableOp│
)batch_normalization_6/AssignMovingAvg/subSub<batch_normalization_6/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_6/moments/Squeeze:output:0*
T0*A
_class7
53loc:@batch_normalization_6/AssignMovingAvg/23020929*
_output_shapes
:2+
)batch_normalization_6/AssignMovingAvg/subк
)batch_normalization_6/AssignMovingAvg/mulMul-batch_normalization_6/AssignMovingAvg/sub:z:04batch_normalization_6/AssignMovingAvg/decay:output:0*
T0*A
_class7
53loc:@batch_normalization_6/AssignMovingAvg/23020929*
_output_shapes
:2+
)batch_normalization_6/AssignMovingAvg/mulЙ
9batch_normalization_6/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp.batch_normalization_6_assignmovingavg_23020929-batch_normalization_6/AssignMovingAvg/mul:z:05^batch_normalization_6/AssignMovingAvg/ReadVariableOp*A
_class7
53loc:@batch_normalization_6/AssignMovingAvg/23020929*
_output_shapes
 *
dtype02;
9batch_normalization_6/AssignMovingAvg/AssignSubVariableOpш
-batch_normalization_6/AssignMovingAvg_1/decayConst*C
_class9
75loc:@batch_normalization_6/AssignMovingAvg_1/23020935*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2/
-batch_normalization_6/AssignMovingAvg_1/decay▌
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOpReadVariableOp0batch_normalization_6_assignmovingavg_1_23020935*
_output_shapes
:*
dtype028
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp╜
+batch_normalization_6/AssignMovingAvg_1/subSub>batch_normalization_6/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_6/moments/Squeeze_1:output:0*
T0*C
_class9
75loc:@batch_normalization_6/AssignMovingAvg_1/23020935*
_output_shapes
:2-
+batch_normalization_6/AssignMovingAvg_1/sub┤
+batch_normalization_6/AssignMovingAvg_1/mulMul/batch_normalization_6/AssignMovingAvg_1/sub:z:06batch_normalization_6/AssignMovingAvg_1/decay:output:0*
T0*C
_class9
75loc:@batch_normalization_6/AssignMovingAvg_1/23020935*
_output_shapes
:2-
+batch_normalization_6/AssignMovingAvg_1/mulХ
;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp0batch_normalization_6_assignmovingavg_1_23020935/batch_normalization_6/AssignMovingAvg_1/mul:z:07^batch_normalization_6/AssignMovingAvg_1/ReadVariableOp*C
_class9
75loc:@batch_normalization_6/AssignMovingAvg_1/23020935*
_output_shapes
 *
dtype02=
;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOpУ
%batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_6/batchnorm/add/y┌
#batch_normalization_6/batchnorm/addAddV20batch_normalization_6/moments/Squeeze_1:output:0.batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_6/batchnorm/addе
%batch_normalization_6/batchnorm/RsqrtRsqrt'batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_6/batchnorm/Rsqrtр
2batch_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization_6/batchnorm/mul/ReadVariableOp▌
#batch_normalization_6/batchnorm/mulMul)batch_normalization_6/batchnorm/Rsqrt:y:0:batch_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_6/batchnorm/mul╪
%batch_normalization_6/batchnorm/mul_1Mulconv3d_6/Elu:activations:0'batch_normalization_6/batchnorm/mul:z:0*
T0*3
_output_shapes!
:         2'
%batch_normalization_6/batchnorm/mul_1╙
%batch_normalization_6/batchnorm/mul_2Mul.batch_normalization_6/moments/Squeeze:output:0'batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_6/batchnorm/mul_2╘
.batch_normalization_6/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_6_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype020
.batch_normalization_6/batchnorm/ReadVariableOp┘
#batch_normalization_6/batchnorm/subSub6batch_normalization_6/batchnorm/ReadVariableOp:value:0)batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_6/batchnorm/subщ
%batch_normalization_6/batchnorm/add_1AddV2)batch_normalization_6/batchnorm/mul_1:z:0'batch_normalization_6/batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2'
%batch_normalization_6/batchnorm/add_1┤
conv3d_7/Conv3D/ReadVariableOpReadVariableOp'conv3d_7_conv3d_readvariableop_resource**
_output_shapes
:*
dtype02 
conv3d_7/Conv3D/ReadVariableOpч
conv3d_7/Conv3DConv3D)batch_normalization_6/batchnorm/add_1:z:0&conv3d_7/Conv3D/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         *
paddingVALID*
strides	
2
conv3d_7/Conv3Dз
conv3d_7/BiasAdd/ReadVariableOpReadVariableOp(conv3d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv3d_7/BiasAdd/ReadVariableOp░
conv3d_7/BiasAddBiasAddconv3d_7/Conv3D:output:0'conv3d_7/BiasAdd/ReadVariableOp:value:0*
T0*3
_output_shapes!
:         2
conv3d_7/BiasAdd|
conv3d_7/EluEluconv3d_7/BiasAdd:output:0*
T0*3
_output_shapes!
:         2
conv3d_7/Elu┼
4batch_normalization_7/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             26
4batch_normalization_7/moments/mean/reduction_indicesё
"batch_normalization_7/moments/meanMeanconv3d_7/Elu:activations:0=batch_normalization_7/moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2$
"batch_normalization_7/moments/mean╩
*batch_normalization_7/moments/StopGradientStopGradient+batch_normalization_7/moments/mean:output:0*
T0**
_output_shapes
:2,
*batch_normalization_7/moments/StopGradientЖ
/batch_normalization_7/moments/SquaredDifferenceSquaredDifferenceconv3d_7/Elu:activations:03batch_normalization_7/moments/StopGradient:output:0*
T0*3
_output_shapes!
:         21
/batch_normalization_7/moments/SquaredDifference═
8batch_normalization_7/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2:
8batch_normalization_7/moments/variance/reduction_indicesЦ
&batch_normalization_7/moments/varianceMean3batch_normalization_7/moments/SquaredDifference:z:0Abatch_normalization_7/moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2(
&batch_normalization_7/moments/variance┼
%batch_normalization_7/moments/SqueezeSqueeze+batch_normalization_7/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2'
%batch_normalization_7/moments/Squeeze═
'batch_normalization_7/moments/Squeeze_1Squeeze/batch_normalization_7/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2)
'batch_normalization_7/moments/Squeeze_1т
+batch_normalization_7/AssignMovingAvg/decayConst*A
_class7
53loc:@batch_normalization_7/AssignMovingAvg/23020968*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2-
+batch_normalization_7/AssignMovingAvg/decay╫
4batch_normalization_7/AssignMovingAvg/ReadVariableOpReadVariableOp.batch_normalization_7_assignmovingavg_23020968*
_output_shapes
:*
dtype026
4batch_normalization_7/AssignMovingAvg/ReadVariableOp│
)batch_normalization_7/AssignMovingAvg/subSub<batch_normalization_7/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_7/moments/Squeeze:output:0*
T0*A
_class7
53loc:@batch_normalization_7/AssignMovingAvg/23020968*
_output_shapes
:2+
)batch_normalization_7/AssignMovingAvg/subк
)batch_normalization_7/AssignMovingAvg/mulMul-batch_normalization_7/AssignMovingAvg/sub:z:04batch_normalization_7/AssignMovingAvg/decay:output:0*
T0*A
_class7
53loc:@batch_normalization_7/AssignMovingAvg/23020968*
_output_shapes
:2+
)batch_normalization_7/AssignMovingAvg/mulЙ
9batch_normalization_7/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp.batch_normalization_7_assignmovingavg_23020968-batch_normalization_7/AssignMovingAvg/mul:z:05^batch_normalization_7/AssignMovingAvg/ReadVariableOp*A
_class7
53loc:@batch_normalization_7/AssignMovingAvg/23020968*
_output_shapes
 *
dtype02;
9batch_normalization_7/AssignMovingAvg/AssignSubVariableOpш
-batch_normalization_7/AssignMovingAvg_1/decayConst*C
_class9
75loc:@batch_normalization_7/AssignMovingAvg_1/23020974*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2/
-batch_normalization_7/AssignMovingAvg_1/decay▌
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOpReadVariableOp0batch_normalization_7_assignmovingavg_1_23020974*
_output_shapes
:*
dtype028
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp╜
+batch_normalization_7/AssignMovingAvg_1/subSub>batch_normalization_7/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_7/moments/Squeeze_1:output:0*
T0*C
_class9
75loc:@batch_normalization_7/AssignMovingAvg_1/23020974*
_output_shapes
:2-
+batch_normalization_7/AssignMovingAvg_1/sub┤
+batch_normalization_7/AssignMovingAvg_1/mulMul/batch_normalization_7/AssignMovingAvg_1/sub:z:06batch_normalization_7/AssignMovingAvg_1/decay:output:0*
T0*C
_class9
75loc:@batch_normalization_7/AssignMovingAvg_1/23020974*
_output_shapes
:2-
+batch_normalization_7/AssignMovingAvg_1/mulХ
;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp0batch_normalization_7_assignmovingavg_1_23020974/batch_normalization_7/AssignMovingAvg_1/mul:z:07^batch_normalization_7/AssignMovingAvg_1/ReadVariableOp*C
_class9
75loc:@batch_normalization_7/AssignMovingAvg_1/23020974*
_output_shapes
 *
dtype02=
;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOpУ
%batch_normalization_7/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_7/batchnorm/add/y┌
#batch_normalization_7/batchnorm/addAddV20batch_normalization_7/moments/Squeeze_1:output:0.batch_normalization_7/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_7/batchnorm/addе
%batch_normalization_7/batchnorm/RsqrtRsqrt'batch_normalization_7/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_7/batchnorm/Rsqrtр
2batch_normalization_7/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_7_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization_7/batchnorm/mul/ReadVariableOp▌
#batch_normalization_7/batchnorm/mulMul)batch_normalization_7/batchnorm/Rsqrt:y:0:batch_normalization_7/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_7/batchnorm/mul╪
%batch_normalization_7/batchnorm/mul_1Mulconv3d_7/Elu:activations:0'batch_normalization_7/batchnorm/mul:z:0*
T0*3
_output_shapes!
:         2'
%batch_normalization_7/batchnorm/mul_1╙
%batch_normalization_7/batchnorm/mul_2Mul.batch_normalization_7/moments/Squeeze:output:0'batch_normalization_7/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_7/batchnorm/mul_2╘
.batch_normalization_7/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_7_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype020
.batch_normalization_7/batchnorm/ReadVariableOp┘
#batch_normalization_7/batchnorm/subSub6batch_normalization_7/batchnorm/ReadVariableOp:value:0)batch_normalization_7/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_7/batchnorm/subщ
%batch_normalization_7/batchnorm/add_1AddV2)batch_normalization_7/batchnorm/mul_1:z:0'batch_normalization_7/batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2'
%batch_normalization_7/batchnorm/add_1Є
average_pooling3d_1/AvgPool3D	AvgPool3D)batch_normalization_7/batchnorm/add_1:z:0*
T0*3
_output_shapes!
:         *
ksize	
*
paddingVALID*
strides	
2
average_pooling3d_1/AvgPool3Ds
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
flatten_1/Constж
flatten_1/ReshapeReshape&average_pooling3d_1/AvgPool3D:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:         А
2
flatten_1/Reshapew
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU╒?2
dropout_3/dropout/Constж
dropout_3/dropout/MulMulflatten_1/Reshape:output:0 dropout_3/dropout/Const:output:0*
T0*(
_output_shapes
:         А
2
dropout_3/dropout/Mul|
dropout_3/dropout/ShapeShapeflatten_1/Reshape:output:0*
T0*
_output_shapes
:2
dropout_3/dropout/Shape╙
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*(
_output_shapes
:         А
*
dtype020
.dropout_3/dropout/random_uniform/RandomUniformЙ
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠>2"
 dropout_3/dropout/GreaterEqual/yч
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А
2 
dropout_3/dropout/GreaterEqualЮ
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А
2
dropout_3/dropout/Castг
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*(
_output_shapes
:         А
2
dropout_3/dropout/Mul_1x
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis┐
concatenate_1/concatConcatV2dropout_3/dropout/Mul_1:z:0inputs_1"concatenate_1/concat/axis:output:0*
N*
T0*(
_output_shapes
:         Г
2
concatenate_1/concatд
layer1/MatMul/ReadVariableOpReadVariableOp%layer1_matmul_readvariableop_resource* 
_output_shapes
:
Г
╚*
dtype02
layer1/MatMul/ReadVariableOpа
layer1/MatMulMatMulconcatenate_1/concat:output:0$layer1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
layer1/MatMulв
layer1/BiasAdd/ReadVariableOpReadVariableOp&layer1_biasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype02
layer1/BiasAdd/ReadVariableOpЮ
layer1/BiasAddBiasAddlayer1/MatMul:product:0%layer1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
layer1/BiasAddk

layer1/EluElulayer1/BiasAdd:output:0*
T0*(
_output_shapes
:         ╚2

layer1/Eluw
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout_4/dropout/Constд
dropout_4/dropout/MulMullayer1/Elu:activations:0 dropout_4/dropout/Const:output:0*
T0*(
_output_shapes
:         ╚2
dropout_4/dropout/Mulz
dropout_4/dropout/ShapeShapelayer1/Elu:activations:0*
T0*
_output_shapes
:2
dropout_4/dropout/Shape╙
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*(
_output_shapes
:         ╚*
dtype020
.dropout_4/dropout/random_uniform/RandomUniformЙ
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2"
 dropout_4/dropout/GreaterEqual/yч
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ╚2 
dropout_4/dropout/GreaterEqualЮ
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ╚2
dropout_4/dropout/Castг
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*(
_output_shapes
:         ╚2
dropout_4/dropout/Mul_1г
layer2/MatMul/ReadVariableOpReadVariableOp%layer2_matmul_readvariableop_resource*
_output_shapes
:	╚*
dtype02
layer2/MatMul/ReadVariableOpЭ
layer2/MatMulMatMuldropout_4/dropout/Mul_1:z:0$layer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
layer2/MatMulб
layer2/BiasAdd/ReadVariableOpReadVariableOp&layer2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
layer2/BiasAdd/ReadVariableOpЭ
layer2/BiasAddBiasAddlayer2/MatMul:product:0%layer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
layer2/BiasAddj

layer2/EluElulayer2/BiasAdd:output:0*
T0*'
_output_shapes
:         2

layer2/Eluw
dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?2
dropout_5/dropout/Constг
dropout_5/dropout/MulMullayer2/Elu:activations:0 dropout_5/dropout/Const:output:0*
T0*'
_output_shapes
:         2
dropout_5/dropout/Mulz
dropout_5/dropout/ShapeShapelayer2/Elu:activations:0*
T0*
_output_shapes
:2
dropout_5/dropout/Shape╥
.dropout_5/dropout/random_uniform/RandomUniformRandomUniform dropout_5/dropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype020
.dropout_5/dropout/random_uniform/RandomUniformЙ
 dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2"
 dropout_5/dropout/GreaterEqual/yц
dropout_5/dropout/GreaterEqualGreaterEqual7dropout_5/dropout/random_uniform/RandomUniform:output:0)dropout_5/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         2 
dropout_5/dropout/GreaterEqualЭ
dropout_5/dropout/CastCast"dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         2
dropout_5/dropout/Castв
dropout_5/dropout/Mul_1Muldropout_5/dropout/Mul:z:0dropout_5/dropout/Cast:y:0*
T0*'
_output_shapes
:         2
dropout_5/dropout/Mul_1е
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_1/MatMul/ReadVariableOpа
dense_1/MatMulMatMuldropout_5/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_1/MatMulд
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOpб
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_1/BiasAddy
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_1/Sigmoid╧
IdentityIdentitydense_1/Sigmoid:y:0:^batch_normalization_4/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp:^batch_normalization_5/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp:^batch_normalization_6/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp:^batch_normalization_7/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*└
_input_shapesо
л:         н:         ::::::::::::::::::::::::::::::2v
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_5/AssignMovingAvg/AssignSubVariableOp9batch_normalization_5/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_6/AssignMovingAvg/AssignSubVariableOp9batch_normalization_6/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_7/AssignMovingAvg/AssignSubVariableOp9batch_normalization_7/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp:^ Z
4
_output_shapes"
 :         н
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
р
└
&__inference_signature_wrapper_23020833
input_3
input_4
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28
identityИвStatefulPartitionedCall╛
StatefulPartitionedCallStatefulPartitionedCallinput_3input_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28*+
Tin$
"2 *
Tout
2*'
_output_shapes
:         *@
_read_only_resource_inputs"
 	
*-
config_proto

CPU

GPU2*0J 8*,
f'R%
#__inference__wrapped_model_230189832
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*└
_input_shapesо
л:         н:         ::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
4
_output_shapes"
 :         н
!
_user_specified_name	input_3:PL
'
_output_shapes
:         
!
_user_specified_name	input_4:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
М
╞
*__inference_model_1_layer_call_fn_23021300
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28
identityИвStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28*+
Tin$
"2 *
Tout
2*'
_output_shapes
:         *@
_read_only_resource_inputs"
 	
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_model_1_layer_call_and_return_conditional_losses_230205502
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*└
_input_shapesо
л:         н:         ::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :         н
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
щ,
╨
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_23019100

inputs
assignmovingavg_23019075
assignmovingavg_1_23019081)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpЩ
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indicesЫ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/meanИ
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient╦
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*N
_output_shapes<
::8                                    2
moments/SquaredDifferenceб
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices╛
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/varianceГ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЛ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1а
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/23019075*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayХ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_23019075*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp┼
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/23019075*
_output_shapes
:2
AssignMovingAvg/sub╝
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/23019075*
_output_shapes
:2
AssignMovingAvg/mulЕ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_23019075AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/23019075*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpж
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/23019081*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЫ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_23019081*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╧
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/23019081*
_output_shapes
:2
AssignMovingAvg_1/sub╞
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/23019081*
_output_shapes
:2
AssignMovingAvg_1/mulС
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_23019081AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/23019081*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulЭ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subм
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/add_1▄
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8                                    ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
╬
e
G__inference_dropout_4_layer_call_and_return_conditional_losses_23022188

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         ╚2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ╚2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:         ╚:P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs
╨^
Ж
E__inference_model_1_layer_call_and_return_conditional_losses_23020234
input_3
input_4
conv3d_4_23019647
conv3d_4_23019649"
batch_normalization_4_23019734"
batch_normalization_4_23019736"
batch_normalization_4_23019738"
batch_normalization_4_23019740
conv3d_5_23019743
conv3d_5_23019745"
batch_normalization_5_23019830"
batch_normalization_5_23019832"
batch_normalization_5_23019834"
batch_normalization_5_23019836
conv3d_6_23019839
conv3d_6_23019841"
batch_normalization_6_23019926"
batch_normalization_6_23019928"
batch_normalization_6_23019930"
batch_normalization_6_23019932
conv3d_7_23019935
conv3d_7_23019937"
batch_normalization_7_23020022"
batch_normalization_7_23020024"
batch_normalization_7_23020026"
batch_normalization_7_23020028
layer1_23020114
layer1_23020116
layer2_23020171
layer2_23020173
dense_1_23020228
dense_1_23020230
identityИв-batch_normalization_4/StatefulPartitionedCallв-batch_normalization_5/StatefulPartitionedCallв-batch_normalization_6/StatefulPartitionedCallв-batch_normalization_7/StatefulPartitionedCallв conv3d_4/StatefulPartitionedCallв conv3d_5/StatefulPartitionedCallв conv3d_6/StatefulPartitionedCallв conv3d_7/StatefulPartitionedCallвdense_1/StatefulPartitionedCallв!dropout_3/StatefulPartitionedCallв!dropout_4/StatefulPartitionedCallв!dropout_5/StatefulPartitionedCallвlayer1/StatefulPartitionedCallвlayer2/StatefulPartitionedCallИ
 conv3d_4/StatefulPartitionedCallStatefulPartitionedCallinput_3conv3d_4_23019647conv3d_4_23019649*
Tin
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_conv3d_4_layer_call_and_return_conditional_losses_230189942"
 conv3d_4/StatefulPartitionedCallн
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv3d_4/StatefulPartitionedCall:output:0batch_normalization_4_23019734batch_normalization_4_23019736batch_normalization_4_23019738batch_normalization_4_23019740*
Tin	
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_230196872/
-batch_normalization_4/StatefulPartitionedCall╖
 conv3d_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0conv3d_5_23019743conv3d_5_23019745*
Tin
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_conv3d_5_layer_call_and_return_conditional_losses_230191562"
 conv3d_5/StatefulPartitionedCallн
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv3d_5/StatefulPartitionedCall:output:0batch_normalization_5_23019830batch_normalization_5_23019832batch_normalization_5_23019834batch_normalization_5_23019836*
Tin	
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_230197832/
-batch_normalization_5/StatefulPartitionedCall╖
 conv3d_6/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0conv3d_6_23019839conv3d_6_23019841*
Tin
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_conv3d_6_layer_call_and_return_conditional_losses_230193182"
 conv3d_6/StatefulPartitionedCallн
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv3d_6/StatefulPartitionedCall:output:0batch_normalization_6_23019926batch_normalization_6_23019928batch_normalization_6_23019930batch_normalization_6_23019932*
Tin	
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_230198792/
-batch_normalization_6/StatefulPartitionedCall╖
 conv3d_7/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0conv3d_7_23019935conv3d_7_23019937*
Tin
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_conv3d_7_layer_call_and_return_conditional_losses_230194802"
 conv3d_7/StatefulPartitionedCallн
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall)conv3d_7/StatefulPartitionedCall:output:0batch_normalization_7_23020022batch_normalization_7_23020024batch_normalization_7_23020026batch_normalization_7_23020028*
Tin	
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_230199752/
-batch_normalization_7/StatefulPartitionedCallФ
#average_pooling3d_1/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*3
_output_shapes!
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*Z
fURS
Q__inference_average_pooling3d_1_layer_call_and_return_conditional_losses_230196362%
#average_pooling3d_1/PartitionedCallс
flatten_1/PartitionedCallPartitionedCall,average_pooling3d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:         А
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_flatten_1_layer_call_and_return_conditional_losses_230200382
flatten_1/PartitionedCallя
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:         А
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_dropout_3_layer_call_and_return_conditional_losses_230200582#
!dropout_3/StatefulPartitionedCallї
concatenate_1/PartitionedCallPartitionedCall*dropout_3/StatefulPartitionedCall:output:0input_4*
Tin
2*
Tout
2*(
_output_shapes
:         Г
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*T
fORM
K__inference_concatenate_1_layer_call_and_return_conditional_losses_230200832
concatenate_1/PartitionedCallТ
layer1/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0layer1_23020114layer1_23020116*
Tin
2*
Tout
2*(
_output_shapes
:         ╚*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_layer1_layer_call_and_return_conditional_losses_230201032 
layer1/StatefulPartitionedCallШ
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall'layer1/StatefulPartitionedCall:output:0"^dropout_3/StatefulPartitionedCall*
Tin
2*
Tout
2*(
_output_shapes
:         ╚* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_dropout_4_layer_call_and_return_conditional_losses_230201312#
!dropout_4/StatefulPartitionedCallХ
layer2/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0layer2_23020171layer2_23020173*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_layer2_layer_call_and_return_conditional_losses_230201602 
layer2/StatefulPartitionedCallЧ
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall'layer2/StatefulPartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_dropout_5_layer_call_and_return_conditional_losses_230201882#
!dropout_5/StatefulPartitionedCallЪ
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0dense_1_23020228dense_1_23020230*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_230202172!
dense_1/StatefulPartitionedCallШ
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall!^conv3d_4/StatefulPartitionedCall!^conv3d_5/StatefulPartitionedCall!^conv3d_6/StatefulPartitionedCall!^conv3d_7/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall^layer1/StatefulPartitionedCall^layer2/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*└
_input_shapesо
л:         н:         ::::::::::::::::::::::::::::::2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2D
 conv3d_4/StatefulPartitionedCall conv3d_4/StatefulPartitionedCall2D
 conv3d_5/StatefulPartitionedCall conv3d_5/StatefulPartitionedCall2D
 conv3d_6/StatefulPartitionedCall conv3d_6/StatefulPartitionedCall2D
 conv3d_7/StatefulPartitionedCall conv3d_7/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2@
layer1/StatefulPartitionedCalllayer1/StatefulPartitionedCall2@
layer2/StatefulPartitionedCalllayer2/StatefulPartitionedCall:] Y
4
_output_shapes"
 :         н
!
_user_specified_name	input_3:PL
'
_output_shapes
:         
!
_user_specified_name	input_4:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
√
~
)__inference_layer2_layer_call_fn_23022218

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall╒
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_layer2_layer_call_and_return_conditional_losses_230201602
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ╚::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
░	
л
8__inference_batch_normalization_7_layer_call_fn_23022018

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*N
_output_shapes<
::8                                    *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_230196192
StatefulPartitionedCall╡
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8                                    ::::22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
М
f
G__inference_dropout_4_layer_call_and_return_conditional_losses_23022183

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         ╚2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╡
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         ╚*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ╚2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ╚2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         ╚2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         ╚2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ╚:P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs
Ь
А
+__inference_conv3d_4_layer_call_fn_23019004

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall■
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*N
_output_shapes<
::8                                    *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_conv3d_4_layer_call_and_return_conditional_losses_230189942
StatefulPartitionedCall╡
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:9                                    н::22
StatefulPartitionedCallStatefulPartitionedCall:w s
O
_output_shapes=
;:9                                    н
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Г
f
G__inference_dropout_5_layer_call_and_return_conditional_losses_23020188

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8О?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2
dropout/GreaterEqual/y╛
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Ъ
А
+__inference_conv3d_7_layer_call_fn_23019490

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall■
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*N
_output_shapes<
::8                                    *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_conv3d_7_layer_call_and_return_conditional_losses_230194802
StatefulPartitionedCall╡
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*U
_input_shapesD
B:8                                    ::22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
░	
л
8__inference_batch_normalization_6_layer_call_fn_23021818

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*N
_output_shapes<
::8                                    *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_230194572
StatefulPartitionedCall╡
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8                                    ::::22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
─
л
8__inference_batch_normalization_4_layer_call_fn_23021500

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallК
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*3
_output_shapes!
:         *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_230197072
StatefulPartitionedCallЪ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:         ::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
о	
л
8__inference_batch_normalization_7_layer_call_fn_23022005

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*N
_output_shapes<
::8                                    *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_230195862
StatefulPartitionedCall╡
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8                                    ::::22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
╟+
╨
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_23021654

inputs
assignmovingavg_23021629
assignmovingavg_1_23021635)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpЩ
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indicesЫ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/meanИ
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient░
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*3
_output_shapes!
:         2
moments/SquaredDifferenceб
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices╛
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/varianceГ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЛ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1а
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/23021629*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayХ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_23021629*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp┼
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/23021629*
_output_shapes
:2
AssignMovingAvg/sub╝
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/23021629*
_output_shapes
:2
AssignMovingAvg/mulЕ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_23021629AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/23021629*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpж
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/23021635*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЫ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_23021635*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╧
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/23021635*
_output_shapes
:2
AssignMovingAvg_1/sub╞
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/23021635*
_output_shapes
:2
AssignMovingAvg_1/mulС
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_23021635AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/23021635*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulВ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:         2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subС
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2
batchnorm/add_1┴
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:         ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
М
f
G__inference_dropout_3_layer_call_and_return_conditional_losses_23020058

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU╒?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         А
2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╡
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А
*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠>2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А
2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А
2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         А
2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         А
2

Identity"
identityIdentity:output:0*'
_input_shapes
:         А
:P L
(
_output_shapes
:         А

 
_user_specified_nameinputs
╬
m
Q__inference_average_pooling3d_1_layer_call_and_return_conditional_losses_23019636

inputs
identity╦
	AvgPool3D	AvgPool3Dinputs*
T0*W
_output_shapesE
C:A                                             *
ksize	
*
paddingVALID*
strides	
2
	AvgPool3DЦ
IdentityIdentityAvgPool3D:output:0*
T0*W
_output_shapesE
C:A                                             2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:A                                             : {
W
_output_shapesE
C:A                                             
 
_user_specified_nameinputs
┬
л
8__inference_batch_normalization_6_layer_call_fn_23021887

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallИ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_230198792
StatefulPartitionedCallЪ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:         ::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
┬
Ц
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_23019457

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulЭ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subм
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/add_1О
IdentityIdentitybatchnorm/add_1:z:0*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8                                    :::::v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
┬
л
8__inference_batch_normalization_4_layer_call_fn_23021487

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallИ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*3
_output_shapes!
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_230196872
StatefulPartitionedCallЪ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:         ::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
┬
Ц
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_23019619

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulЭ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subм
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/add_1О
IdentityIdentitybatchnorm/add_1:z:0*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8                                    :::::v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
╩
e
G__inference_dropout_5_layer_call_and_return_conditional_losses_23020193

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:         2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Д
╞
*__inference_model_1_layer_call_fn_23021234
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28
identityИвStatefulPartitionedCall┌
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28*+
Tin$
"2 *
Tout
2*'
_output_shapes
:         *8
_read_only_resource_inputs
	*-
config_proto

CPU

GPU2*0J 8*N
fIRG
E__inference_model_1_layer_call_and_return_conditional_losses_230204022
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*└
_input_shapesо
л:         н:         ::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :         н
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
─
л
8__inference_batch_normalization_6_layer_call_fn_23021900

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallК
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*3
_output_shapes!
:         *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_230198992
StatefulPartitionedCallЪ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:         ::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
║
Ц
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_23019899

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulВ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:         2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subС
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:         :::::[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
╟+
╨
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_23019687

inputs
assignmovingavg_23019662
assignmovingavg_1_23019668)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpЩ
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indicesЫ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/meanИ
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient░
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*3
_output_shapes!
:         2
moments/SquaredDifferenceб
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices╛
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/varianceГ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЛ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1а
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/23019662*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayХ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_23019662*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp┼
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/23019662*
_output_shapes
:2
AssignMovingAvg/sub╝
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/23019662*
_output_shapes
:2
AssignMovingAvg/mulЕ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_23019662AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/23019662*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpж
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/23019668*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЫ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_23019668*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╧
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/23019668*
_output_shapes
:2
AssignMovingAvg_1/sub╞
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/23019668*
_output_shapes
:2
AssignMovingAvg_1/mulС
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_23019668AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/23019668*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulВ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:         2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subС
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2
batchnorm/add_1┴
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:         ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
╝
u
K__inference_concatenate_1_layer_call_and_return_conditional_losses_23020083

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisА
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:         Г
2
concatd
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:         Г
2

Identity"
identityIdentity:output:0*:
_input_shapes)
':         А
:         :P L
(
_output_shapes
:         А

 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs
░	
л
8__inference_batch_normalization_5_layer_call_fn_23021618

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*N
_output_shapes<
::8                                    *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_230192952
StatefulPartitionedCall╡
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8                                    ::::22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
щ,
╨
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_23019262

inputs
assignmovingavg_23019237
assignmovingavg_1_23019243)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpЩ
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indicesЫ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/meanИ
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient╦
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*N
_output_shapes<
::8                                    2
moments/SquaredDifferenceб
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices╛
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/varianceГ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЛ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1а
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/23019237*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayХ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_23019237*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp┼
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/23019237*
_output_shapes
:2
AssignMovingAvg/sub╝
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/23019237*
_output_shapes
:2
AssignMovingAvg/mulЕ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_23019237AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/23019237*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpж
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/23019243*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЫ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_23019243*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╧
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/23019243*
_output_shapes
:2
AssignMovingAvg_1/sub╞
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/23019243*
_output_shapes
:2
AssignMovingAvg_1/mulС
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_23019243AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/23019243*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulЭ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subм
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/add_1▄
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8                                    ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
щ,
╨
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_23021972

inputs
assignmovingavg_23021947
assignmovingavg_1_23021953)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpЩ
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indicesЫ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/meanИ
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient╦
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*N
_output_shapes<
::8                                    2
moments/SquaredDifferenceб
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices╛
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/varianceГ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЛ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1а
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/23021947*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayХ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_23021947*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp┼
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/23021947*
_output_shapes
:2
AssignMovingAvg/sub╝
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/23021947*
_output_shapes
:2
AssignMovingAvg/mulЕ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_23021947AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/23021947*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpж
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/23021953*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЫ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_23021953*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╧
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/23021953*
_output_shapes
:2
AssignMovingAvg_1/sub╞
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/23021953*
_output_shapes
:2
AssignMovingAvg_1/mulС
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_23021953AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/23021953*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulЭ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subм
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/add_1▄
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8                                    ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
М
f
G__inference_dropout_4_layer_call_and_return_conditional_losses_23020131

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         ╚2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╡
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         ╚*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ╚2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ╚2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         ╚2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         ╚2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ╚:P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs
ф
м
D__inference_layer2_layer_call_and_return_conditional_losses_23022209

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	╚*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddU
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:         2
Elue
IdentityIdentityElu:activations:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ╚:::P L
(
_output_shapes
:         ╚
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
─
л
8__inference_batch_normalization_7_layer_call_fn_23022100

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallК
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*3
_output_shapes!
:         *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_230199952
StatefulPartitionedCallЪ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:         ::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
о	
л
8__inference_batch_normalization_4_layer_call_fn_23021405

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*N
_output_shapes<
::8                                    *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*\
fWRU
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_230191002
StatefulPartitionedCall╡
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8                                    ::::22
StatefulPartitionedCallStatefulPartitionedCall:v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
╟
c
G__inference_flatten_1_layer_call_and_return_conditional_losses_23022106

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         А
2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         А
2

Identity"
identityIdentity:output:0*2
_input_shapes!
:         :[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs
║
Ц
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_23019803

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulВ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*3
_output_shapes!
:         2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subС
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*3
_output_shapes!
:         2
batchnorm/add_1s
IdentityIdentitybatchnorm/add_1:z:0*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:         :::::[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ъ
м
D__inference_layer1_layer_call_and_return_conditional_losses_23022162

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Г
╚*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:╚*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╚2	
BiasAddV
EluEluBiasAdd:output:0*
T0*(
_output_shapes
:         ╚2
Eluf
IdentityIdentityElu:activations:0*
T0*(
_output_shapes
:         ╚2

Identity"
identityIdentity:output:0*/
_input_shapes
:         Г
:::P L
(
_output_shapes
:         Г

 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
╟
c
G__inference_flatten_1_layer_call_and_return_conditional_losses_23020038

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         А
2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         А
2

Identity"
identityIdentity:output:0*2
_input_shapes!
:         :[ W
3
_output_shapes!
:         
 
_user_specified_nameinputs
щ,
╨
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_23021772

inputs
assignmovingavg_23021747
assignmovingavg_1_23021753)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpЩ
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2 
moments/mean/reduction_indicesЫ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/meanИ
moments/StopGradientStopGradientmoments/mean:output:0*
T0**
_output_shapes
:2
moments/StopGradient╦
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*N
_output_shapes<
::8                                    2
moments/SquaredDifferenceб
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"moments/variance/reduction_indices╛
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0**
_output_shapes
:*
	keep_dims(2
moments/varianceГ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЛ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1а
AssignMovingAvg/decayConst*+
_class!
loc:@AssignMovingAvg/23021747*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayХ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_23021747*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp┼
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*+
_class!
loc:@AssignMovingAvg/23021747*
_output_shapes
:2
AssignMovingAvg/sub╝
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg/23021747*
_output_shapes
:2
AssignMovingAvg/mulЕ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_23021747AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/23021747*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpж
AssignMovingAvg_1/decayConst*-
_class#
!loc:@AssignMovingAvg_1/23021753*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЫ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_23021753*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╧
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/23021753*
_output_shapes
:2
AssignMovingAvg_1/sub╞
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/23021753*
_output_shapes
:2
AssignMovingAvg_1/mulС
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_23021753AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/23021753*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulЭ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subм
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/add_1▄
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8                                    ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Й
e
,__inference_dropout_3_layer_call_fn_23022133

inputs
identityИвStatefulPartitionedCall┐
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:         А
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*P
fKRI
G__inference_dropout_3_layer_call_and_return_conditional_losses_230200582
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А
2

Identity"
identityIdentity:output:0*'
_input_shapes
:         А
22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А

 
_user_specified_nameinputs
И

о
F__inference_conv3d_4_layer_call_and_return_conditional_losses_23018994

inputs"
conv3d_readvariableop_resource#
biasadd_readvariableop_resource
identityИЪ
Conv3D/ReadVariableOpReadVariableOpconv3d_readvariableop_resource*+
_output_shapes
:н*
dtype02
Conv3D/ReadVariableOp─
Conv3DConv3DinputsConv3D/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8                                    *
paddingVALID*
strides	
2
Conv3DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpз
BiasAddBiasAddConv3D:output:0BiasAdd/ReadVariableOp:value:0*
T0*N
_output_shapes<
::8                                    2	
BiasAddЛ
IdentityIdentityBiasAdd:output:0*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:9                                    н:::w s
O
_output_shapes=
;:9                                    н
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
┬
Ц
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_23021992

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulЭ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subм
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*N
_output_shapes<
::8                                    2
batchnorm/add_1О
IdentityIdentitybatchnorm/add_1:z:0*
T0*N
_output_shapes<
::8                                    2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:8                                    :::::v r
N
_output_shapes<
::8                                    
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
цй
╚$
!__inference__traced_save_23022536
file_prefix.
*savev2_conv3d_4_kernel_read_readvariableop,
(savev2_conv3d_4_bias_read_readvariableop:
6savev2_batch_normalization_4_gamma_read_readvariableop9
5savev2_batch_normalization_4_beta_read_readvariableop@
<savev2_batch_normalization_4_moving_mean_read_readvariableopD
@savev2_batch_normalization_4_moving_variance_read_readvariableop.
*savev2_conv3d_5_kernel_read_readvariableop,
(savev2_conv3d_5_bias_read_readvariableop:
6savev2_batch_normalization_5_gamma_read_readvariableop9
5savev2_batch_normalization_5_beta_read_readvariableop@
<savev2_batch_normalization_5_moving_mean_read_readvariableopD
@savev2_batch_normalization_5_moving_variance_read_readvariableop.
*savev2_conv3d_6_kernel_read_readvariableop,
(savev2_conv3d_6_bias_read_readvariableop:
6savev2_batch_normalization_6_gamma_read_readvariableop9
5savev2_batch_normalization_6_beta_read_readvariableop@
<savev2_batch_normalization_6_moving_mean_read_readvariableopD
@savev2_batch_normalization_6_moving_variance_read_readvariableop.
*savev2_conv3d_7_kernel_read_readvariableop,
(savev2_conv3d_7_bias_read_readvariableop:
6savev2_batch_normalization_7_gamma_read_readvariableop9
5savev2_batch_normalization_7_beta_read_readvariableop@
<savev2_batch_normalization_7_moving_mean_read_readvariableopD
@savev2_batch_normalization_7_moving_variance_read_readvariableop.
*savev2_layer1_1_kernel_read_readvariableop,
(savev2_layer1_1_bias_read_readvariableop.
*savev2_layer2_1_kernel_read_readvariableop,
(savev2_layer2_1_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_conv3d_4_kernel_m_read_readvariableop3
/savev2_adam_conv3d_4_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_4_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_4_beta_m_read_readvariableop5
1savev2_adam_conv3d_5_kernel_m_read_readvariableop3
/savev2_adam_conv3d_5_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_5_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_5_beta_m_read_readvariableop5
1savev2_adam_conv3d_6_kernel_m_read_readvariableop3
/savev2_adam_conv3d_6_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_6_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_6_beta_m_read_readvariableop5
1savev2_adam_conv3d_7_kernel_m_read_readvariableop3
/savev2_adam_conv3d_7_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_7_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_7_beta_m_read_readvariableop5
1savev2_adam_layer1_1_kernel_m_read_readvariableop3
/savev2_adam_layer1_1_bias_m_read_readvariableop5
1savev2_adam_layer2_1_kernel_m_read_readvariableop3
/savev2_adam_layer2_1_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop5
1savev2_adam_conv3d_4_kernel_v_read_readvariableop3
/savev2_adam_conv3d_4_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_4_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_4_beta_v_read_readvariableop5
1savev2_adam_conv3d_5_kernel_v_read_readvariableop3
/savev2_adam_conv3d_5_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_5_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_5_beta_v_read_readvariableop5
1savev2_adam_conv3d_6_kernel_v_read_readvariableop3
/savev2_adam_conv3d_6_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_6_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_6_beta_v_read_readvariableop5
1savev2_adam_conv3d_7_kernel_v_read_readvariableop3
/savev2_adam_conv3d_7_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_7_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_7_beta_v_read_readvariableop5
1savev2_adam_layer1_1_kernel_v_read_readvariableop3
/savev2_adam_layer1_1_bias_v_read_readvariableop5
1savev2_adam_layer2_1_kernel_v_read_readvariableop3
/savev2_adam_layer2_1_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop
savev2_1_const

identity_1ИвMergeV2CheckpointsвSaveV2вSaveV2_1П
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
ConstН
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_0e344457dfe841beb5ee320efc592887/part2	
Const_1Л
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename─-
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:Q*
dtype0*╓,
value╠,B╔,QB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_namesн
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:Q*
dtype0*╖
valueнBкQB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices№"
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv3d_4_kernel_read_readvariableop(savev2_conv3d_4_bias_read_readvariableop6savev2_batch_normalization_4_gamma_read_readvariableop5savev2_batch_normalization_4_beta_read_readvariableop<savev2_batch_normalization_4_moving_mean_read_readvariableop@savev2_batch_normalization_4_moving_variance_read_readvariableop*savev2_conv3d_5_kernel_read_readvariableop(savev2_conv3d_5_bias_read_readvariableop6savev2_batch_normalization_5_gamma_read_readvariableop5savev2_batch_normalization_5_beta_read_readvariableop<savev2_batch_normalization_5_moving_mean_read_readvariableop@savev2_batch_normalization_5_moving_variance_read_readvariableop*savev2_conv3d_6_kernel_read_readvariableop(savev2_conv3d_6_bias_read_readvariableop6savev2_batch_normalization_6_gamma_read_readvariableop5savev2_batch_normalization_6_beta_read_readvariableop<savev2_batch_normalization_6_moving_mean_read_readvariableop@savev2_batch_normalization_6_moving_variance_read_readvariableop*savev2_conv3d_7_kernel_read_readvariableop(savev2_conv3d_7_bias_read_readvariableop6savev2_batch_normalization_7_gamma_read_readvariableop5savev2_batch_normalization_7_beta_read_readvariableop<savev2_batch_normalization_7_moving_mean_read_readvariableop@savev2_batch_normalization_7_moving_variance_read_readvariableop*savev2_layer1_1_kernel_read_readvariableop(savev2_layer1_1_bias_read_readvariableop*savev2_layer2_1_kernel_read_readvariableop(savev2_layer2_1_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_conv3d_4_kernel_m_read_readvariableop/savev2_adam_conv3d_4_bias_m_read_readvariableop=savev2_adam_batch_normalization_4_gamma_m_read_readvariableop<savev2_adam_batch_normalization_4_beta_m_read_readvariableop1savev2_adam_conv3d_5_kernel_m_read_readvariableop/savev2_adam_conv3d_5_bias_m_read_readvariableop=savev2_adam_batch_normalization_5_gamma_m_read_readvariableop<savev2_adam_batch_normalization_5_beta_m_read_readvariableop1savev2_adam_conv3d_6_kernel_m_read_readvariableop/savev2_adam_conv3d_6_bias_m_read_readvariableop=savev2_adam_batch_normalization_6_gamma_m_read_readvariableop<savev2_adam_batch_normalization_6_beta_m_read_readvariableop1savev2_adam_conv3d_7_kernel_m_read_readvariableop/savev2_adam_conv3d_7_bias_m_read_readvariableop=savev2_adam_batch_normalization_7_gamma_m_read_readvariableop<savev2_adam_batch_normalization_7_beta_m_read_readvariableop1savev2_adam_layer1_1_kernel_m_read_readvariableop/savev2_adam_layer1_1_bias_m_read_readvariableop1savev2_adam_layer2_1_kernel_m_read_readvariableop/savev2_adam_layer2_1_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop1savev2_adam_conv3d_4_kernel_v_read_readvariableop/savev2_adam_conv3d_4_bias_v_read_readvariableop=savev2_adam_batch_normalization_4_gamma_v_read_readvariableop<savev2_adam_batch_normalization_4_beta_v_read_readvariableop1savev2_adam_conv3d_5_kernel_v_read_readvariableop/savev2_adam_conv3d_5_bias_v_read_readvariableop=savev2_adam_batch_normalization_5_gamma_v_read_readvariableop<savev2_adam_batch_normalization_5_beta_v_read_readvariableop1savev2_adam_conv3d_6_kernel_v_read_readvariableop/savev2_adam_conv3d_6_bias_v_read_readvariableop=savev2_adam_batch_normalization_6_gamma_v_read_readvariableop<savev2_adam_batch_normalization_6_beta_v_read_readvariableop1savev2_adam_conv3d_7_kernel_v_read_readvariableop/savev2_adam_conv3d_7_bias_v_read_readvariableop=savev2_adam_batch_normalization_7_gamma_v_read_readvariableop<savev2_adam_batch_normalization_7_beta_v_read_readvariableop1savev2_adam_layer1_1_kernel_v_read_readvariableop/savev2_adam_layer1_1_bias_v_read_readvariableop1savev2_adam_layer2_1_kernel_v_read_readvariableop/savev2_adam_layer2_1_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *_
dtypesU
S2Q	2
SaveV2Г
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardм
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1в
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_namesО
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices╧
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1у
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesм
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

IdentityБ

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*╓
_input_shapes─
┴: :н::::::::::::::::::::::::
Г
╚:╚:	╚:::: : : : : : : :н::::::::::::::::
Г
╚:╚:	╚::::н::::::::::::::::
Г
╚:╚:	╚:::: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:1-
+
_output_shapes
:н: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
:: 	

_output_shapes
:: 


_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::0,
*
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::&"
 
_output_shapes
:
Г
╚:!

_output_shapes	
:╚:%!

_output_shapes
:	╚: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :1&-
+
_output_shapes
:н: '

_output_shapes
:: (

_output_shapes
:: )

_output_shapes
::0*,
*
_output_shapes
:: +

_output_shapes
:: ,

_output_shapes
:: -

_output_shapes
::0.,
*
_output_shapes
:: /

_output_shapes
:: 0

_output_shapes
:: 1

_output_shapes
::02,
*
_output_shapes
:: 3

_output_shapes
:: 4

_output_shapes
:: 5

_output_shapes
::&6"
 
_output_shapes
:
Г
╚:!7

_output_shapes	
:╚:%8!

_output_shapes
:	╚: 9

_output_shapes
::$: 

_output_shapes

:: ;

_output_shapes
::1<-
+
_output_shapes
:н: =

_output_shapes
:: >

_output_shapes
:: ?

_output_shapes
::0@,
*
_output_shapes
:: A

_output_shapes
:: B

_output_shapes
:: C

_output_shapes
::0D,
*
_output_shapes
:: E

_output_shapes
:: F

_output_shapes
:: G

_output_shapes
::0H,
*
_output_shapes
:: I

_output_shapes
:: J

_output_shapes
:: K

_output_shapes
::&L"
 
_output_shapes
:
Г
╚:!M

_output_shapes	
:╚:%N!

_output_shapes
:	╚: O

_output_shapes
::$P 

_output_shapes

:: Q

_output_shapes
::R

_output_shapes
: "пL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ї
serving_defaultр
H
input_3=
serving_default_input_3:0         н
;
input_40
serving_default_input_4:0         ;
dense_10
StatefulPartitionedCall:0         tensorflow/serving/predict:▓В
иж
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer_with_weights-8
layer-14
layer-15
layer_with_weights-9
layer-16
layer-17
layer_with_weights-10
layer-18
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
+Р&call_and_return_all_conditional_losses
С_default_save_signature
Т__call__"║а
_tf_keras_modelЯа{"class_name": "Model", "name": "model_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 24, 24, 24, 173]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "Conv3D", "config": {"name": "conv3d_4", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 24, 24, 24, 173]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [1, 1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_4", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_4", "inbound_nodes": [[["conv3d_4", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_5", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 24, 24, 24, 20]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_5", "inbound_nodes": [[["batch_normalization_4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_5", "inbound_nodes": [[["conv3d_5", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_6", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 22, 22, 22, 20]}, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [4, 4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_6", "inbound_nodes": [[["batch_normalization_5", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_6", "inbound_nodes": [[["conv3d_6", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_7", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 19, 19, 19, 30]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [4, 4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_7", "inbound_nodes": [[["batch_normalization_6", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_7", "inbound_nodes": [[["conv3d_7", 0, 0, {}]]]}, {"class_name": "AveragePooling3D", "config": {"name": "average_pooling3d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [4, 4, 4]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [4, 4, 4]}, "data_format": "channels_last"}, "name": "average_pooling3d_1", "inbound_nodes": [[["batch_normalization_7", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["average_pooling3d_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}, "name": "dropout_3", "inbound_nodes": [[["flatten_1", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_1", "inbound_nodes": [[["dropout_3", 0, 0, {}], ["input_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "layer1", "trainable": true, "dtype": "float32", "units": 200, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 4, "axis": 0}}, "bias_constraint": {"class_name": "MaxNorm", "config": {"max_value": 4, "axis": 0}}}, "name": "layer1", "inbound_nodes": [[["concatenate_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_4", "inbound_nodes": [[["layer1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "layer2", "trainable": true, "dtype": "float32", "units": 20, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 4, "axis": 0}}, "bias_constraint": {"class_name": "MaxNorm", "config": {"max_value": 4, "axis": 0}}}, "name": "layer2", "inbound_nodes": [[["dropout_4", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_5", "inbound_nodes": [[["layer2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dropout_5", 0, 0, {}]]]}], "input_layers": [["input_3", 0, 0], ["input_4", 0, 0]], "output_layers": [["dense_1", 0, 0]]}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 24, 24, 24, 173]}, {"class_name": "TensorShape", "items": [null, 3]}], "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 24, 24, 24, 173]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "Conv3D", "config": {"name": "conv3d_4", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 24, 24, 24, 173]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [1, 1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_4", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_4", "inbound_nodes": [[["conv3d_4", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_5", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 24, 24, 24, 20]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_5", "inbound_nodes": [[["batch_normalization_4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_5", "inbound_nodes": [[["conv3d_5", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_6", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 22, 22, 22, 20]}, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [4, 4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_6", "inbound_nodes": [[["batch_normalization_5", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_6", "inbound_nodes": [[["conv3d_6", 0, 0, {}]]]}, {"class_name": "Conv3D", "config": {"name": "conv3d_7", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 19, 19, 19, 30]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [4, 4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3d_7", "inbound_nodes": [[["batch_normalization_6", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_7", "inbound_nodes": [[["conv3d_7", 0, 0, {}]]]}, {"class_name": "AveragePooling3D", "config": {"name": "average_pooling3d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [4, 4, 4]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [4, 4, 4]}, "data_format": "channels_last"}, "name": "average_pooling3d_1", "inbound_nodes": [[["batch_normalization_7", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["average_pooling3d_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}, "name": "dropout_3", "inbound_nodes": [[["flatten_1", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_1", "inbound_nodes": [[["dropout_3", 0, 0, {}], ["input_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "layer1", "trainable": true, "dtype": "float32", "units": 200, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 4, "axis": 0}}, "bias_constraint": {"class_name": "MaxNorm", "config": {"max_value": 4, "axis": 0}}}, "name": "layer1", "inbound_nodes": [[["concatenate_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_4", "inbound_nodes": [[["layer1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "layer2", "trainable": true, "dtype": "float32", "units": 20, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 4, "axis": 0}}, "bias_constraint": {"class_name": "MaxNorm", "config": {"max_value": 4, "axis": 0}}}, "name": "layer2", "inbound_nodes": [[["dropout_4", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_5", "inbound_nodes": [[["layer2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dropout_5", 0, 0, {}]]]}], "input_layers": [["input_3", 0, 0], ["input_4", 0, 0]], "output_layers": [["dense_1", 0, 0]]}}, "training_config": {"loss": "binary_crossentropy", "metrics": null, "weighted_metrics": null, "loss_weights": null, "sample_weight_mode": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
Е"В
_tf_keras_input_layerт{"class_name": "InputLayer", "name": "input_3", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 24, 24, 24, 173]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 24, 24, 24, 173]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}}
м

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
+У&call_and_return_all_conditional_losses
Ф__call__"Е

_tf_keras_layerы	{"class_name": "Conv3D", "name": "conv3d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 24, 24, 24, 173]}, "stateful": false, "config": {"name": "conv3d_4", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 24, 24, 24, 173]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [1, 1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"-1": 173}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 24, 24, 173]}}
Э	
 axis
	!gamma
"beta
#moving_mean
$moving_variance
%trainable_variables
&regularization_losses
'	variables
(	keras_api
+Х&call_and_return_all_conditional_losses
Ц__call__"╟
_tf_keras_layerн{"class_name": "BatchNormalization", "name": "batch_normalization_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"4": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 24, 24, 20]}}
е

)kernel
*bias
+trainable_variables
,regularization_losses
-	variables
.	keras_api
+Ч&call_and_return_all_conditional_losses
Ш__call__"■	
_tf_keras_layerф	{"class_name": "Conv3D", "name": "conv3d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 24, 24, 24, 20]}, "stateful": false, "config": {"name": "conv3d_5", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 24, 24, 24, 20]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [3, 3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 24, 24, 20]}}
Э	
/axis
	0gamma
1beta
2moving_mean
3moving_variance
4trainable_variables
5regularization_losses
6	variables
7	keras_api
+Щ&call_and_return_all_conditional_losses
Ъ__call__"╟
_tf_keras_layerн{"class_name": "BatchNormalization", "name": "batch_normalization_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"4": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 22, 22, 22, 20]}}
е

8kernel
9bias
:trainable_variables
;regularization_losses
<	variables
=	keras_api
+Ы&call_and_return_all_conditional_losses
Ь__call__"■	
_tf_keras_layerф	{"class_name": "Conv3D", "name": "conv3d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 22, 22, 22, 20]}, "stateful": false, "config": {"name": "conv3d_6", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 22, 22, 22, 20]}, "dtype": "float32", "filters": 30, "kernel_size": {"class_name": "__tuple__", "items": [4, 4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 22, 22, 22, 20]}}
Э	
>axis
	?gamma
@beta
Amoving_mean
Bmoving_variance
Ctrainable_variables
Dregularization_losses
E	variables
F	keras_api
+Э&call_and_return_all_conditional_losses
Ю__call__"╟
_tf_keras_layerн{"class_name": "BatchNormalization", "name": "batch_normalization_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"4": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 19, 19, 19, 30]}}
е

Gkernel
Hbias
Itrainable_variables
Jregularization_losses
K	variables
L	keras_api
+Я&call_and_return_all_conditional_losses
а__call__"■	
_tf_keras_layerф	{"class_name": "Conv3D", "name": "conv3d_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 19, 19, 19, 30]}, "stateful": false, "config": {"name": "conv3d_7", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 19, 19, 19, 30]}, "dtype": "float32", "filters": 20, "kernel_size": {"class_name": "__tuple__", "items": [4, 4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1, 1]}, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"-1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 19, 19, 19, 30]}}
Э	
Maxis
	Ngamma
Obeta
Pmoving_mean
Qmoving_variance
Rtrainable_variables
Sregularization_losses
T	variables
U	keras_api
+б&call_and_return_all_conditional_losses
в__call__"╟
_tf_keras_layerн{"class_name": "BatchNormalization", "name": "batch_normalization_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [4], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {"4": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 16, 20]}}
Ё
Vtrainable_variables
Wregularization_losses
X	variables
Y	keras_api
+г&call_and_return_all_conditional_losses
д__call__"▀
_tf_keras_layer┼{"class_name": "AveragePooling3D", "name": "average_pooling3d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "average_pooling3d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [4, 4, 4]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [4, 4, 4]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {}}}}
┼
Ztrainable_variables
[regularization_losses
\	variables
]	keras_api
+е&call_and_return_all_conditional_losses
ж__call__"┤
_tf_keras_layerЪ{"class_name": "Flatten", "name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
─
^trainable_variables
_regularization_losses
`	variables
a	keras_api
+з&call_and_return_all_conditional_losses
и__call__"│
_tf_keras_layerЩ{"class_name": "Dropout", "name": "dropout_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}
щ"ц
_tf_keras_input_layer╞{"class_name": "InputLayer", "name": "input_4", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}}
н
btrainable_variables
cregularization_losses
d	variables
e	keras_api
+й&call_and_return_all_conditional_losses
к__call__"Ь
_tf_keras_layerВ{"class_name": "Concatenate", "name": "concatenate_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1280]}, {"class_name": "TensorShape", "items": [null, 3]}]}
╔

fkernel
gbias
htrainable_variables
iregularization_losses
j	variables
k	keras_api
+л&call_and_return_all_conditional_losses
м__call__"в
_tf_keras_layerИ{"class_name": "Dense", "name": "layer1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "layer1", "trainable": true, "dtype": "float32", "units": 200, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 4, "axis": 0}}, "bias_constraint": {"class_name": "MaxNorm", "config": {"max_value": 4, "axis": 0}}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1283}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1283]}}
─
ltrainable_variables
mregularization_losses
n	variables
o	keras_api
+н&call_and_return_all_conditional_losses
о__call__"│
_tf_keras_layerЩ{"class_name": "Dropout", "name": "dropout_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
╞

pkernel
qbias
rtrainable_variables
sregularization_losses
t	variables
u	keras_api
+п&call_and_return_all_conditional_losses
░__call__"Я
_tf_keras_layerЕ{"class_name": "Dense", "name": "layer2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "layer2", "trainable": true, "dtype": "float32", "units": 20, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": {"class_name": "MaxNorm", "config": {"max_value": 4, "axis": 0}}, "bias_constraint": {"class_name": "MaxNorm", "config": {"max_value": 4, "axis": 0}}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 200}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 200]}}
─
vtrainable_variables
wregularization_losses
x	variables
y	keras_api
+▒&call_and_return_all_conditional_losses
▓__call__"│
_tf_keras_layerЩ{"class_name": "Dropout", "name": "dropout_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
╤

zkernel
{bias
|trainable_variables
}regularization_losses
~	variables
	keras_api
+│&call_and_return_all_conditional_losses
┤__call__"к
_tf_keras_layerР{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20]}}
Р
	Аiter
Бbeta_1
Вbeta_2

Гdecay
Дlearning_ratemфmх!mц"mч)mш*mщ0mъ1mы8mь9mэ?mю@mяGmЁHmёNmЄOmєfmЇgmїpmЎqmўzm°{m∙v·v√!v№"v¤)v■*v 0vА1vБ8vВ9vГ?vД@vЕGvЖHvЗNvИOvЙfvКgvЛpvМqvНzvО{vП"
	optimizer
╞
0
1
!2
"3
)4
*5
06
17
88
99
?10
@11
G12
H13
N14
O15
f16
g17
p18
q19
z20
{21"
trackable_list_wrapper
 "
trackable_list_wrapper
Ж
0
1
!2
"3
#4
$5
)6
*7
08
19
210
311
812
913
?14
@15
A16
B17
G18
H19
N20
O21
P22
Q23
f24
g25
p26
q27
z28
{29"
trackable_list_wrapper
╙
Еlayer_metrics
Жnon_trainable_variables
Зlayers
Иmetrics
trainable_variables
regularization_losses
 Йlayer_regularization_losses
	variables
Т__call__
С_default_save_signature
+Р&call_and_return_all_conditional_losses
'Р"call_and_return_conditional_losses"
_generic_user_object
-
╡serving_default"
signature_map
.:,н2conv3d_4/kernel
:2conv3d_4/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
╡
Кlayer_metrics
Лnon_trainable_variables
Мmetrics
Нlayers
trainable_variables
regularization_losses
 Оlayer_regularization_losses
	variables
Ф__call__
+У&call_and_return_all_conditional_losses
'У"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2batch_normalization_4/gamma
(:&2batch_normalization_4/beta
1:/ (2!batch_normalization_4/moving_mean
5:3 (2%batch_normalization_4/moving_variance
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
!0
"1
#2
$3"
trackable_list_wrapper
╡
Пlayer_metrics
Рnon_trainable_variables
Сmetrics
Тlayers
%trainable_variables
&regularization_losses
 Уlayer_regularization_losses
'	variables
Ц__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses"
_generic_user_object
-:+2conv3d_5/kernel
:2conv3d_5/bias
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
╡
Фlayer_metrics
Хnon_trainable_variables
Цmetrics
Чlayers
+trainable_variables
,regularization_losses
 Шlayer_regularization_losses
-	variables
Ш__call__
+Ч&call_and_return_all_conditional_losses
'Ч"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2batch_normalization_5/gamma
(:&2batch_normalization_5/beta
1:/ (2!batch_normalization_5/moving_mean
5:3 (2%batch_normalization_5/moving_variance
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
<
00
11
22
33"
trackable_list_wrapper
╡
Щlayer_metrics
Ъnon_trainable_variables
Ыmetrics
Ьlayers
4trainable_variables
5regularization_losses
 Эlayer_regularization_losses
6	variables
Ъ__call__
+Щ&call_and_return_all_conditional_losses
'Щ"call_and_return_conditional_losses"
_generic_user_object
-:+2conv3d_6/kernel
:2conv3d_6/bias
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
╡
Юlayer_metrics
Яnon_trainable_variables
аmetrics
бlayers
:trainable_variables
;regularization_losses
 вlayer_regularization_losses
<	variables
Ь__call__
+Ы&call_and_return_all_conditional_losses
'Ы"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2batch_normalization_6/gamma
(:&2batch_normalization_6/beta
1:/ (2!batch_normalization_6/moving_mean
5:3 (2%batch_normalization_6/moving_variance
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
?0
@1
A2
B3"
trackable_list_wrapper
╡
гlayer_metrics
дnon_trainable_variables
еmetrics
жlayers
Ctrainable_variables
Dregularization_losses
 зlayer_regularization_losses
E	variables
Ю__call__
+Э&call_and_return_all_conditional_losses
'Э"call_and_return_conditional_losses"
_generic_user_object
-:+2conv3d_7/kernel
:2conv3d_7/bias
.
G0
H1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
╡
иlayer_metrics
йnon_trainable_variables
кmetrics
лlayers
Itrainable_variables
Jregularization_losses
 мlayer_regularization_losses
K	variables
а__call__
+Я&call_and_return_all_conditional_losses
'Я"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2batch_normalization_7/gamma
(:&2batch_normalization_7/beta
1:/ (2!batch_normalization_7/moving_mean
5:3 (2%batch_normalization_7/moving_variance
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
N0
O1
P2
Q3"
trackable_list_wrapper
╡
нlayer_metrics
оnon_trainable_variables
пmetrics
░layers
Rtrainable_variables
Sregularization_losses
 ▒layer_regularization_losses
T	variables
в__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
▓layer_metrics
│non_trainable_variables
┤metrics
╡layers
Vtrainable_variables
Wregularization_losses
 ╢layer_regularization_losses
X	variables
д__call__
+г&call_and_return_all_conditional_losses
'г"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
╖layer_metrics
╕non_trainable_variables
╣metrics
║layers
Ztrainable_variables
[regularization_losses
 ╗layer_regularization_losses
\	variables
ж__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
╝layer_metrics
╜non_trainable_variables
╛metrics
┐layers
^trainable_variables
_regularization_losses
 └layer_regularization_losses
`	variables
и__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
┴layer_metrics
┬non_trainable_variables
├metrics
─layers
btrainable_variables
cregularization_losses
 ┼layer_regularization_losses
d	variables
к__call__
+й&call_and_return_all_conditional_losses
'й"call_and_return_conditional_losses"
_generic_user_object
#:!
Г
╚2layer1_1/kernel
:╚2layer1_1/bias
.
f0
g1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
╡
╞layer_metrics
╟non_trainable_variables
╚metrics
╔layers
htrainable_variables
iregularization_losses
 ╩layer_regularization_losses
j	variables
м__call__
+л&call_and_return_all_conditional_losses
'л"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
╦layer_metrics
╠non_trainable_variables
═metrics
╬layers
ltrainable_variables
mregularization_losses
 ╧layer_regularization_losses
n	variables
о__call__
+н&call_and_return_all_conditional_losses
'н"call_and_return_conditional_losses"
_generic_user_object
": 	╚2layer2_1/kernel
:2layer2_1/bias
.
p0
q1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
p0
q1"
trackable_list_wrapper
╡
╨layer_metrics
╤non_trainable_variables
╥metrics
╙layers
rtrainable_variables
sregularization_losses
 ╘layer_regularization_losses
t	variables
░__call__
+п&call_and_return_all_conditional_losses
'п"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
╒layer_metrics
╓non_trainable_variables
╫metrics
╪layers
vtrainable_variables
wregularization_losses
 ┘layer_regularization_losses
x	variables
▓__call__
+▒&call_and_return_all_conditional_losses
'▒"call_and_return_conditional_losses"
_generic_user_object
 :2dense_1/kernel
:2dense_1/bias
.
z0
{1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
╡
┌layer_metrics
█non_trainable_variables
▄metrics
▌layers
|trainable_variables
}regularization_losses
 ▐layer_regularization_losses
~	variables
┤__call__
+│&call_and_return_all_conditional_losses
'│"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_dict_wrapper
X
#0
$1
22
33
A4
B5
P6
Q7"
trackable_list_wrapper
о
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18"
trackable_list_wrapper
(
▀0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
┐

рtotal

сcount
т	variables
у	keras_api"Д
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
0
р0
с1"
trackable_list_wrapper
.
т	variables"
_generic_user_object
3:1н2Adam/conv3d_4/kernel/m
 :2Adam/conv3d_4/bias/m
.:,2"Adam/batch_normalization_4/gamma/m
-:+2!Adam/batch_normalization_4/beta/m
2:02Adam/conv3d_5/kernel/m
 :2Adam/conv3d_5/bias/m
.:,2"Adam/batch_normalization_5/gamma/m
-:+2!Adam/batch_normalization_5/beta/m
2:02Adam/conv3d_6/kernel/m
 :2Adam/conv3d_6/bias/m
.:,2"Adam/batch_normalization_6/gamma/m
-:+2!Adam/batch_normalization_6/beta/m
2:02Adam/conv3d_7/kernel/m
 :2Adam/conv3d_7/bias/m
.:,2"Adam/batch_normalization_7/gamma/m
-:+2!Adam/batch_normalization_7/beta/m
(:&
Г
╚2Adam/layer1_1/kernel/m
!:╚2Adam/layer1_1/bias/m
':%	╚2Adam/layer2_1/kernel/m
 :2Adam/layer2_1/bias/m
%:#2Adam/dense_1/kernel/m
:2Adam/dense_1/bias/m
3:1н2Adam/conv3d_4/kernel/v
 :2Adam/conv3d_4/bias/v
.:,2"Adam/batch_normalization_4/gamma/v
-:+2!Adam/batch_normalization_4/beta/v
2:02Adam/conv3d_5/kernel/v
 :2Adam/conv3d_5/bias/v
.:,2"Adam/batch_normalization_5/gamma/v
-:+2!Adam/batch_normalization_5/beta/v
2:02Adam/conv3d_6/kernel/v
 :2Adam/conv3d_6/bias/v
.:,2"Adam/batch_normalization_6/gamma/v
-:+2!Adam/batch_normalization_6/beta/v
2:02Adam/conv3d_7/kernel/v
 :2Adam/conv3d_7/bias/v
.:,2"Adam/batch_normalization_7/gamma/v
-:+2!Adam/batch_normalization_7/beta/v
(:&
Г
╚2Adam/layer1_1/kernel/v
!:╚2Adam/layer1_1/bias/v
':%	╚2Adam/layer2_1/kernel/v
 :2Adam/layer2_1/bias/v
%:#2Adam/dense_1/kernel/v
:2Adam/dense_1/bias/v
т2▀
E__inference_model_1_layer_call_and_return_conditional_losses_23021043
E__inference_model_1_layer_call_and_return_conditional_losses_23021168
E__inference_model_1_layer_call_and_return_conditional_losses_23020234
E__inference_model_1_layer_call_and_return_conditional_losses_23020316└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ц2У
#__inference__wrapped_model_23018983ы
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *[вX
VЪS
.К+
input_3         н
!К
input_4         
Ў2є
*__inference_model_1_layer_call_fn_23021234
*__inference_model_1_layer_call_fn_23021300
*__inference_model_1_layer_call_fn_23020465
*__inference_model_1_layer_call_fn_23020613└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
│2░
F__inference_conv3d_4_layer_call_and_return_conditional_losses_23018994х
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *EвB
@К=9                                    н
Ш2Х
+__inference_conv3d_4_layer_call_fn_23019004х
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *EвB
@К=9                                    н
О2Л
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_23021392
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_23021454
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_23021474
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_23021372┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
в2Я
8__inference_batch_normalization_4_layer_call_fn_23021418
8__inference_batch_normalization_4_layer_call_fn_23021487
8__inference_batch_normalization_4_layer_call_fn_23021500
8__inference_batch_normalization_4_layer_call_fn_23021405┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
▓2п
F__inference_conv3d_5_layer_call_and_return_conditional_losses_23019156ф
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *DвA
?К<8                                    
Ч2Ф
+__inference_conv3d_5_layer_call_fn_23019166ф
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *DвA
?К<8                                    
О2Л
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_23021592
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_23021674
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_23021572
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_23021654┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
в2Я
8__inference_batch_normalization_5_layer_call_fn_23021700
8__inference_batch_normalization_5_layer_call_fn_23021618
8__inference_batch_normalization_5_layer_call_fn_23021687
8__inference_batch_normalization_5_layer_call_fn_23021605┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
▓2п
F__inference_conv3d_6_layer_call_and_return_conditional_losses_23019318ф
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *DвA
?К<8                                    
Ч2Ф
+__inference_conv3d_6_layer_call_fn_23019328ф
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *DвA
?К<8                                    
О2Л
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_23021772
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_23021792
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_23021874
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_23021854┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
в2Я
8__inference_batch_normalization_6_layer_call_fn_23021900
8__inference_batch_normalization_6_layer_call_fn_23021818
8__inference_batch_normalization_6_layer_call_fn_23021805
8__inference_batch_normalization_6_layer_call_fn_23021887┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
▓2п
F__inference_conv3d_7_layer_call_and_return_conditional_losses_23019480ф
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *DвA
?К<8                                    
Ч2Ф
+__inference_conv3d_7_layer_call_fn_23019490ф
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *DвA
?К<8                                    
О2Л
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_23021972
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_23022054
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_23021992
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_23022074┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
в2Я
8__inference_batch_normalization_7_layer_call_fn_23022018
8__inference_batch_normalization_7_layer_call_fn_23022100
8__inference_batch_normalization_7_layer_call_fn_23022087
8__inference_batch_normalization_7_layer_call_fn_23022005┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╞2├
Q__inference_average_pooling3d_1_layer_call_and_return_conditional_losses_23019636э
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *MвJ
HКEA                                             
л2и
6__inference_average_pooling3d_1_layer_call_fn_23019642э
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *MвJ
HКEA                                             
ё2ю
G__inference_flatten_1_layer_call_and_return_conditional_losses_23022106в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╓2╙
,__inference_flatten_1_layer_call_fn_23022111в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╠2╔
G__inference_dropout_3_layer_call_and_return_conditional_losses_23022123
G__inference_dropout_3_layer_call_and_return_conditional_losses_23022128┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ц2У
,__inference_dropout_3_layer_call_fn_23022138
,__inference_dropout_3_layer_call_fn_23022133┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ї2Є
K__inference_concatenate_1_layer_call_and_return_conditional_losses_23022145в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
┌2╫
0__inference_concatenate_1_layer_call_fn_23022151в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ю2ы
D__inference_layer1_layer_call_and_return_conditional_losses_23022162в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╙2╨
)__inference_layer1_layer_call_fn_23022171в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╠2╔
G__inference_dropout_4_layer_call_and_return_conditional_losses_23022188
G__inference_dropout_4_layer_call_and_return_conditional_losses_23022183┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ц2У
,__inference_dropout_4_layer_call_fn_23022198
,__inference_dropout_4_layer_call_fn_23022193┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ю2ы
D__inference_layer2_layer_call_and_return_conditional_losses_23022209в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╙2╨
)__inference_layer2_layer_call_fn_23022218в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╠2╔
G__inference_dropout_5_layer_call_and_return_conditional_losses_23022230
G__inference_dropout_5_layer_call_and_return_conditional_losses_23022235┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ц2У
,__inference_dropout_5_layer_call_fn_23022240
,__inference_dropout_5_layer_call_fn_23022245┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
я2ь
E__inference_dense_1_layer_call_and_return_conditional_losses_23022256в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╘2╤
*__inference_dense_1_layer_call_fn_23022265в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
<B:
&__inference_signature_wrapper_23020833input_3input_4т
#__inference__wrapped_model_23018983║$!#")*302189B?A@GHQNPOfgpqz{eвb
[вX
VЪS
.К+
input_3         н
!К
input_4         
к "1к.
,
dense_1!К
dense_1         О
Q__inference_average_pooling3d_1_layer_call_and_return_conditional_losses_23019636╕_в\
UвR
PКM
inputsA                                             
к "UвR
KКH
0A                                             
Ъ ц
6__inference_average_pooling3d_1_layer_call_fn_23019642л_в\
UвR
PКM
inputsA                                             
к "HКEA                                             И
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_23021372░#$!"ZвW
PвM
GКD
inputs8                                    
p
к "LвI
BК?
08                                    
Ъ И
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_23021392░$!#"ZвW
PвM
GКD
inputs8                                    
p 
к "LвI
BК?
08                                    
Ъ ╤
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_23021454z#$!"?в<
5в2
,К)
inputs         
p
к "1в.
'К$
0         
Ъ ╤
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_23021474z$!#"?в<
5в2
,К)
inputs         
p 
к "1в.
'К$
0         
Ъ р
8__inference_batch_normalization_4_layer_call_fn_23021405г#$!"ZвW
PвM
GКD
inputs8                                    
p
к "?К<8                                    р
8__inference_batch_normalization_4_layer_call_fn_23021418г$!#"ZвW
PвM
GКD
inputs8                                    
p 
к "?К<8                                    й
8__inference_batch_normalization_4_layer_call_fn_23021487m#$!"?в<
5в2
,К)
inputs         
p
к "$К!         й
8__inference_batch_normalization_4_layer_call_fn_23021500m$!#"?в<
5в2
,К)
inputs         
p 
к "$К!         И
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_23021572░2301ZвW
PвM
GКD
inputs8                                    
p
к "LвI
BК?
08                                    
Ъ И
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_23021592░3021ZвW
PвM
GКD
inputs8                                    
p 
к "LвI
BК?
08                                    
Ъ ╤
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_23021654z2301?в<
5в2
,К)
inputs         
p
к "1в.
'К$
0         
Ъ ╤
S__inference_batch_normalization_5_layer_call_and_return_conditional_losses_23021674z3021?в<
5в2
,К)
inputs         
p 
к "1в.
'К$
0         
Ъ р
8__inference_batch_normalization_5_layer_call_fn_23021605г2301ZвW
PвM
GКD
inputs8                                    
p
к "?К<8                                    р
8__inference_batch_normalization_5_layer_call_fn_23021618г3021ZвW
PвM
GКD
inputs8                                    
p 
к "?К<8                                    й
8__inference_batch_normalization_5_layer_call_fn_23021687m2301?в<
5в2
,К)
inputs         
p
к "$К!         й
8__inference_batch_normalization_5_layer_call_fn_23021700m3021?в<
5в2
,К)
inputs         
p 
к "$К!         И
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_23021772░AB?@ZвW
PвM
GКD
inputs8                                    
p
к "LвI
BК?
08                                    
Ъ И
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_23021792░B?A@ZвW
PвM
GКD
inputs8                                    
p 
к "LвI
BК?
08                                    
Ъ ╤
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_23021854zAB?@?в<
5в2
,К)
inputs         
p
к "1в.
'К$
0         
Ъ ╤
S__inference_batch_normalization_6_layer_call_and_return_conditional_losses_23021874zB?A@?в<
5в2
,К)
inputs         
p 
к "1в.
'К$
0         
Ъ р
8__inference_batch_normalization_6_layer_call_fn_23021805гAB?@ZвW
PвM
GКD
inputs8                                    
p
к "?К<8                                    р
8__inference_batch_normalization_6_layer_call_fn_23021818гB?A@ZвW
PвM
GКD
inputs8                                    
p 
к "?К<8                                    й
8__inference_batch_normalization_6_layer_call_fn_23021887mAB?@?в<
5в2
,К)
inputs         
p
к "$К!         й
8__inference_batch_normalization_6_layer_call_fn_23021900mB?A@?в<
5в2
,К)
inputs         
p 
к "$К!         И
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_23021972░PQNOZвW
PвM
GКD
inputs8                                    
p
к "LвI
BК?
08                                    
Ъ И
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_23021992░QNPOZвW
PвM
GКD
inputs8                                    
p 
к "LвI
BК?
08                                    
Ъ ╤
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_23022054zPQNO?в<
5в2
,К)
inputs         
p
к "1в.
'К$
0         
Ъ ╤
S__inference_batch_normalization_7_layer_call_and_return_conditional_losses_23022074zQNPO?в<
5в2
,К)
inputs         
p 
к "1в.
'К$
0         
Ъ р
8__inference_batch_normalization_7_layer_call_fn_23022005гPQNOZвW
PвM
GКD
inputs8                                    
p
к "?К<8                                    р
8__inference_batch_normalization_7_layer_call_fn_23022018гQNPOZвW
PвM
GКD
inputs8                                    
p 
к "?К<8                                    й
8__inference_batch_normalization_7_layer_call_fn_23022087mPQNO?в<
5в2
,К)
inputs         
p
к "$К!         й
8__inference_batch_normalization_7_layer_call_fn_23022100mQNPO?в<
5в2
,К)
inputs         
p 
к "$К!         ╒
K__inference_concatenate_1_layer_call_and_return_conditional_losses_23022145Е[вX
QвN
LЪI
#К 
inputs/0         А

"К
inputs/1         
к "&в#
К
0         Г

Ъ м
0__inference_concatenate_1_layer_call_fn_23022151x[вX
QвN
LЪI
#К 
inputs/0         А

"К
inputs/1         
к "К         Г
Ў
F__inference_conv3d_4_layer_call_and_return_conditional_losses_23018994лWвT
MвJ
HКE
inputs9                                    н
к "LвI
BК?
08                                    
Ъ ╬
+__inference_conv3d_4_layer_call_fn_23019004ЮWвT
MвJ
HКE
inputs9                                    н
к "?К<8                                    ї
F__inference_conv3d_5_layer_call_and_return_conditional_losses_23019156к)*VвS
LвI
GКD
inputs8                                    
к "LвI
BК?
08                                    
Ъ ═
+__inference_conv3d_5_layer_call_fn_23019166Э)*VвS
LвI
GКD
inputs8                                    
к "?К<8                                    ї
F__inference_conv3d_6_layer_call_and_return_conditional_losses_23019318к89VвS
LвI
GКD
inputs8                                    
к "LвI
BК?
08                                    
Ъ ═
+__inference_conv3d_6_layer_call_fn_23019328Э89VвS
LвI
GКD
inputs8                                    
к "?К<8                                    ї
F__inference_conv3d_7_layer_call_and_return_conditional_losses_23019480кGHVвS
LвI
GКD
inputs8                                    
к "LвI
BК?
08                                    
Ъ ═
+__inference_conv3d_7_layer_call_fn_23019490ЭGHVвS
LвI
GКD
inputs8                                    
к "?К<8                                    е
E__inference_dense_1_layer_call_and_return_conditional_losses_23022256\z{/в,
%в"
 К
inputs         
к "%в"
К
0         
Ъ }
*__inference_dense_1_layer_call_fn_23022265Oz{/в,
%в"
 К
inputs         
к "К         й
G__inference_dropout_3_layer_call_and_return_conditional_losses_23022123^4в1
*в'
!К
inputs         А

p
к "&в#
К
0         А

Ъ й
G__inference_dropout_3_layer_call_and_return_conditional_losses_23022128^4в1
*в'
!К
inputs         А

p 
к "&в#
К
0         А

Ъ Б
,__inference_dropout_3_layer_call_fn_23022133Q4в1
*в'
!К
inputs         А

p
к "К         А
Б
,__inference_dropout_3_layer_call_fn_23022138Q4в1
*в'
!К
inputs         А

p 
к "К         А
й
G__inference_dropout_4_layer_call_and_return_conditional_losses_23022183^4в1
*в'
!К
inputs         ╚
p
к "&в#
К
0         ╚
Ъ й
G__inference_dropout_4_layer_call_and_return_conditional_losses_23022188^4в1
*в'
!К
inputs         ╚
p 
к "&в#
К
0         ╚
Ъ Б
,__inference_dropout_4_layer_call_fn_23022193Q4в1
*в'
!К
inputs         ╚
p
к "К         ╚Б
,__inference_dropout_4_layer_call_fn_23022198Q4в1
*в'
!К
inputs         ╚
p 
к "К         ╚з
G__inference_dropout_5_layer_call_and_return_conditional_losses_23022230\3в0
)в&
 К
inputs         
p
к "%в"
К
0         
Ъ з
G__inference_dropout_5_layer_call_and_return_conditional_losses_23022235\3в0
)в&
 К
inputs         
p 
к "%в"
К
0         
Ъ 
,__inference_dropout_5_layer_call_fn_23022240O3в0
)в&
 К
inputs         
p
к "К         
,__inference_dropout_5_layer_call_fn_23022245O3в0
)в&
 К
inputs         
p 
к "К         ░
G__inference_flatten_1_layer_call_and_return_conditional_losses_23022106e;в8
1в.
,К)
inputs         
к "&в#
К
0         А

Ъ И
,__inference_flatten_1_layer_call_fn_23022111X;в8
1в.
,К)
inputs         
к "К         А
ж
D__inference_layer1_layer_call_and_return_conditional_losses_23022162^fg0в-
&в#
!К
inputs         Г

к "&в#
К
0         ╚
Ъ ~
)__inference_layer1_layer_call_fn_23022171Qfg0в-
&в#
!К
inputs         Г

к "К         ╚е
D__inference_layer2_layer_call_and_return_conditional_losses_23022209]pq0в-
&в#
!К
inputs         ╚
к "%в"
К
0         
Ъ }
)__inference_layer2_layer_call_fn_23022218Ppq0в-
&в#
!К
inputs         ╚
к "К         А
E__inference_model_1_layer_call_and_return_conditional_losses_23020234╢#$!")*230189AB?@GHPQNOfgpqz{mвj
cв`
VЪS
.К+
input_3         н
!К
input_4         
p

 
к "%в"
К
0         
Ъ А
E__inference_model_1_layer_call_and_return_conditional_losses_23020316╢$!#")*302189B?A@GHQNPOfgpqz{mвj
cв`
VЪS
.К+
input_3         н
!К
input_4         
p 

 
к "%в"
К
0         
Ъ В
E__inference_model_1_layer_call_and_return_conditional_losses_23021043╕#$!")*230189AB?@GHPQNOfgpqz{oвl
eвb
XЪU
/К,
inputs/0         н
"К
inputs/1         
p

 
к "%в"
К
0         
Ъ В
E__inference_model_1_layer_call_and_return_conditional_losses_23021168╕$!#")*302189B?A@GHQNPOfgpqz{oвl
eвb
XЪU
/К,
inputs/0         н
"К
inputs/1         
p 

 
к "%в"
К
0         
Ъ ╪
*__inference_model_1_layer_call_fn_23020465й#$!")*230189AB?@GHPQNOfgpqz{mвj
cв`
VЪS
.К+
input_3         н
!К
input_4         
p

 
к "К         ╪
*__inference_model_1_layer_call_fn_23020613й$!#")*302189B?A@GHQNPOfgpqz{mвj
cв`
VЪS
.К+
input_3         н
!К
input_4         
p 

 
к "К         ┌
*__inference_model_1_layer_call_fn_23021234л#$!")*230189AB?@GHPQNOfgpqz{oвl
eвb
XЪU
/К,
inputs/0         н
"К
inputs/1         
p

 
к "К         ┌
*__inference_model_1_layer_call_fn_23021300л$!#")*302189B?A@GHQNPOfgpqz{oвl
eвb
XЪU
/К,
inputs/0         н
"К
inputs/1         
p 

 
к "К         Ў
&__inference_signature_wrapper_23020833╦$!#")*302189B?A@GHQNPOfgpqz{vвs
в 
lкi
9
input_3.К+
input_3         н
,
input_4!К
input_4         "1к.
,
dense_1!К
dense_1         