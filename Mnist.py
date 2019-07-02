# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

"""
# %matplotlib inline  #have no idea why it doesn't work on script

a = numpy.zeros([3,2])
a[0,0] = 1
a[0,1] = 2
a[1,0]=9
a[2,1]=12
print(a)
matplotlib.pyplot.imshow(a, interpolation = "nearest")
"""

#주석은 이렇게 샵으로 달아버립니다. ㅇㅋㄷㅋ?


import numpy, scipy.special#, matplotlib.pyplot
#시각화가 외부 윈도우가 아닌 현재의 노트북 내에서 보이도록 설정
#%matplotlib.pyplot


class neuralNetwork:
    
    #신경망 초기화
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        #가중치 행렬 wih 와 who
        #배열 내 가중치는 w_i_j로 표기. 노드 i에서 다음 계층의 노드 j로 연결됨을 의미
        #w11 w21
        #w12 w22 등
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        #활성화 함수로 시그모이드 함수를 이용
        
        self.lr  = learningrate
        
        #활성화 함수로는 시그모이드 함수를 이용        
        self.activation_function = lambda x: scipy.special.expit(x)

        pass
    
    #numpy.random.rand(3,3)

    #신경망 학습시키기
    def train(self, inputs_list, targets_list):
        #임력 리스트를 2차원의 행렬로 변환
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        #은닉 계층으로 들어오는 신호를 계산
        hidden_inputs = numpy.dot(self.wih, inputs)
        #은닉 계층에서 나가는 신호를 계산
        hidden_outputs = self.activation_function(hidden_inputs)
        
        #최종 출력 계층으로 들어오는 신호를 계산
        final_inputs = numpy.dot(self.who, hidden_outputs)
        
        #최종출력 계층에서 나가는 신호를 계산
        final_outputs = self.activation_function(final_inputs)
        
        #오차
        output_errors = targets - final_outputs
        
        #은닉 계층의 오차는 가중치에 의해 나뉜 출력 계층의 오차들을 재조합해 계산
        hidden_errors = numpy.dot(self.who.T, output_errors)
        
        #은닉 계층과 출력 계층 간의 가중치 업데이트
        self.who += self.lr*numpy.dot((output_errors*final_outputs*(1.0 - final_outputs)),numpy.transpose(hidden_outputs))
        
        #입력 계층과 은닉 계층 간의 가중치 업데이트
        self.wih += self.lr*numpy.dot((hidden_errors*hidden_outputs*(1.0 - hidden_outputs)), numpy.transpose(inputs))
        pass
    
    #신경망에 질의하기
    def query(self, inputs_list):
        #입력리스트를 2차원 행렬로 변환
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        #은닉 계층으로 들어오는 신호를 계산
        hidden_inputs = numpy.dot(self.wih, inputs)        
        
        #은닉 계층에서 나가는 신호를 계산
        hidden_outputs = self.activation_function(hidden_inputs)
        
        #최종출력 계층으로 들어오는 신호를 계산
        final_inputs = numpy.dot(self.who, hidden_outputs)
        
        #최종 출력 계층에서 나가는 신호를 계산
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs

    
input_nodes  = 784
hidden_nodes = 250
output_nodes = 10

learning_rate = 0.2

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

training_data_file = open("C:\\Users\Gabriel\Desktop\mnist_dataset\mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

#신경망 학습시키기

#주기(epoch)란 학습데이터가 학습을 위해 사용되는 횟수를 의미
epoch = 2

for e in range(epoch):
    #학습데이터 모음 내의 모든 레코드 탐색
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
        #결과값 생성(실제값인 0.99 외에는 모두 0.01)
        targets = numpy.zeros(output_nodes)+0.01
        #all_values[0]은 이 레코드에 대한 결과 값[0]은 이 레코드에 대한 결과값
        targets[int(all_values[0])]=0.99
        n.train(inputs, targets)
        pass
    pass

#학습 데이터 모음 내의 모든 레코드 탐색
for record in training_data_list:
    all_values = record.split(',')
    
    inputs = (numpy.asfarray(all_values[1:])/255.0*0.99) + 0.01
   
    #결과 값 생성 (실제 값인 0.99 외에는 모두 0.01)
    targets = numpy.zeros(output_nodes) +0.01
    
    #all_values[0]은 이 레코드에 대한 결과 값
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)
    pass


test_data_file = open("C:\\Users\Gabriel\Desktop\mnist_dataset\mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

#신경망 테스트 

#신경망의 성능의 지표가 되는 성적표를 아무 값도 가지지 않도록 초기화
scorecard= []

#테스트 데이터 모음 내의 모든 레코드 탐색
for record in test_data_list:
    #레코드를 쉼표에 의해 분리
    all_values = record.split(',')
    #정답은 첫 번째 값
    correct_label = int(all_values[0])
    print(correct_label, "correct label")
    #입력 값의 범위와 값 조정
    inputs = (numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
    #신경망에 질의
    outputs = n.query(inputs)
    #가장 높은 값의 인덱스는 레이블의 인덱스와 일치
    label = numpy.argmax(outputs)
    print(label, "network's answer")
    #정답 또는 오답을 리스트에 추가
    if (label == correct_label):
        #정답인 경우 성적표에 1을 더함
        scorecard.append(1)
    else:
        #정답이 아닌경우 0을 더함
        scorecard.append(0)
    
        pass
    pass

# 정답의 비율인 성적을 계산해 출력
scorecard_array = numpy.asarray(scorecard)
print ( "performance = ", scorecard_array.sum()/scorecard_array.size)

        






