#include<bits/stdc++.h>
#include<cstdlib>
#include<cmath>
#include<time.h>
using namespace std;

int Lsum=0,FloatValue_len;

double ActFunc_ReLU(double x){return x>0?x:0;}
double ActFunc_Sigmoid(double x){return 1/(1+exp(-x));}

int RandomV(int Min, int Max){//整数随机函数
    return rand()%(Max+1)+Min;
}
double GetRandomVPro(double Min,double Max,int len=FloatValue_len){//双精度随机函数
    int digit_int=RandomV(floor(Min),ceil(Max)),digit_float=RandomV(0,pow(10,len)-1);
    return double(digit_int)+double(digit_float)/pow(10,len);
}
double GetRandomV(double Min, double Max) {
    return Min + (Max - Min) * (rand() / (RAND_MAX + 1.0));
}
typedef double(*ActFunc)(double);

struct Neuron{
    double* x=nullptr;
    double* y=nullptr;
};

class Layer{
    public:
        ActFunc actfunction;
        int Nsum=-1,Lid=-1;//神经元总数（未初始化，默认为-1），层id（未初始化时id为-1）
        Neuron* Ldata;//动态神经元数组
        double b;//偏置项
        
        /*
        动态数组定义方法
            int n = 10; // 假设n是动态计算得到的数组长度
            int* arr = new int[n]; // 在堆上分配一个大小为n的数组
            // 使用arr
            delete[] arr; // 使用完毕后释放内存
        */
        
        void __init__(int nsum,double bias,ActFunc func){//初始化
            Lid=Lsum+1;//更改层id为总id数+1
            Lsum++;//总层数加1
            actfunction=func;//设置该层使用的激活函数
            Nsum=nsum;//设置神经元总数
            b=bias;//设置偏置项值
            Ldata=new (std::nothrow) Neuron[Nsum];//动态分配神经元总数
            if (Ldata == nullptr) {
            std::cerr << "Error:Memory allocation failed!动态数组内存分配失败！" << std::endl;
                return ; // Exit with error code
            }
            
            if(Lid!=1){
                for (int i=0;i<Nsum;i++){
                    delete Ldata[i].x;
                    Ldata[i].y=new double;
                    
                    *(Ldata[i].y)=0.0;
                }
            }
            else{
                for (int i=0;i<Nsum;i++){
                    delete Ldata[i].y;
                    
                    Ldata[i].x=new double;
                    *(Ldata[i].x)=0.0;
                }
            }//若是输入层，设置每个神经元输入为0
        }
        
        /*
        void resizeArray(int*& arr, int oldSize, int newSize) {
            int* newArr = new int[newSize]; // 分配新的内存空间
            for (int i = 0; i < oldSize; ++i) {
                newArr[i] = arr[i]; // 复制现有元素
            }
            delete[] arr; // 释放旧内存空间
            arr = newArr; // 更新指针
        }
        */
        
        void TypeLayerV(double Input[], int arrsize){
            if(Lid!=1){
                char ctrlkey;
                cout<<"Warning:There seem a wrong piece of code in the Program!程序某段代码似乎出现了错误！"<<endl<<"----Layer(id:"<<Lid<<") is not a Input Layer but the program use the function'typeLayer'! Layerid为"<<Lid<<"的层不是输入层但程序调用了手动输入数据函数！"<<endl<<"Hand-Control:Do you want to continue?手动控制：是否继续？(reply回复 'Y' or 'N')"<<endl;
                cin >> ctrlkey;
                if(tolower(ctrlkey)=='n'){cout<<"Info:The typing has stopped.已结束赋值。"<<endl;return ;}
            }
            for(int i=0;i<arrsize;i++){
                *(Ldata[i].x)=Input[i];
            }
        }
        

        double CostV(double expect[]){
            double sum=0.0;
            for(int i=0;i<Nsum;i++){
                sum+=pow(*(Ldata[i].y)-expect[i],2);
            }
            return sum;
        }
        ~Layer() {
            if (Ldata != nullptr) {
                for (int i = 0; i < Nsum; i++) {
                    delete Ldata[i].x;  // 释放输入指针
                    delete Ldata[i].y;  // 释放输出指针
                }
                delete[] Ldata;  // 释放神经元数组
            }
        }

};

/*int GetDigitLen(int n,int len=2){
    return n<10 ? 1 : (n/10<10 ? len : GetDigitLen(n/10,len+1));
}*/
class Net {
public:
    Layer Lin, *Lhid, Lout;
    int LhidSum = -1;
    double*** Wsums; // 改为三维指针，存储各层连接的权重矩阵

    void __init__(int LinNsum, double LinBias, ActFunc LinActFunc, int Lhidsum, int LhidNsums[], double LhidBiases[], ActFunc LhidActFunc[], int LoutNsum, double LoutBias, ActFunc LoutActFunc) {
        Lin.__init__(LinNsum, LinBias, LinActFunc);
        if (Lhidsum != 0) {
            Lhid = new Layer[Lhidsum];
            for (int i = 0; i < Lhidsum; i++) {
                Lhid[i].__init__(LhidNsums[i], LhidBiases[i], LhidActFunc[i]);
            }
        }
        Lout.__init__(LoutNsum, LoutBias, LoutActFunc);
        LhidSum = Lhidsum;
    }

    void ConnectWbyRandomV() {
        // 权重矩阵数量：输入层到隐藏层 + 隐藏层之间 + 隐藏层到输出层
        int numConnections = LhidSum + 1;
        Wsums = new double**[numConnections];

        // 输入层到第一个隐藏层
        if (LhidSum >= 1) {
            int inputSize = Lin.Nsum;
            int hiddenSize = Lhid[0].Nsum;
            Wsums[0] = new double*[inputSize];
            for (int j = 0; j < inputSize; j++) {
                Wsums[0][j] = new double[hiddenSize];
                double heWeight = sqrt(6.0 / (inputSize + hiddenSize)); // 调整He初始化
                for (int k = 0; k < hiddenSize; k++) {
                    Wsums[0][j][k] = GetRandomV(-heWeight, heWeight);
                }
            }
        }

        // 隐藏层之间的连接
        for (int i = 0; i < LhidSum - 1; i++) {
            int currentSize = Lhid[i].Nsum;
            int nextSize = Lhid[i + 1].Nsum;
            Wsums[i + 1] = new double*[currentSize];
            for (int j = 0; j < currentSize; j++) {
                Wsums[i + 1][j] = new double[nextSize];
                double heWeight = sqrt(6.0 / (currentSize + nextSize));
                for (int k = 0; k < nextSize; k++) {
                    Wsums[i + 1][j][k] = GetRandomV(-heWeight, heWeight);
                }
            }
        }

        // 最后一个隐藏层到输出层
        if (LhidSum >= 1) {
            int lastHiddenSize = Lhid[LhidSum - 1].Nsum;
            int outputSize = Lout.Nsum;
            Wsums[LhidSum] = new double*[lastHiddenSize];
            for (int j = 0; j < lastHiddenSize; j++) {
                Wsums[LhidSum][j] = new double[outputSize];
                double heWeight = sqrt(6.0 / (lastHiddenSize + outputSize));
                for (int k = 0; k < outputSize; k++) {
                    Wsums[LhidSum][j][k] = GetRandomV(-heWeight, heWeight);
                }
            }
        }
    }
    
    void EchoNetInfo(){
        if (Wsums == nullptr) {
            cout << "Error: Weights not initialized!" << endl;
            return;
        }
        cout<<"InputLayer -- HiddenLayer[0]:"<<endl;
        for(int j=0;j<Lin.Nsum;j++){
            for(int k=0;k<Lhid[0].Nsum;k++){
                cout<<setw(16)<<"  W("<<j<<'-'<<k<<")"<<Wsums[0][j][k]<<';';
            }
            cout<<endl;
        }
        cout<<"  Bias:"<<Lhid[0].b<<endl;
        for(int i=1;i<LhidSum;i++){
            cout<<endl<<"HiddenLayer["<<i-1<<"] -- HiddenLayer["<<i<<"]:"<<endl;
            for(int j=0;j<Lhid[i-1].Nsum;j++){
                for(int k=0;k<Lhid[i].Nsum;k++){
                    cout<<setw(16)<<"  W("<<j<<'-'<<k<<")"<<Wsums[i][j][k]<<';';
                }
                cout<<endl;
            }
            cout<<"  Bias:"<<Lhid[LhidSum-1].b<<endl;
        }
        cout<<endl<<"HiddenLayer["<<LhidSum-1<<"] -- OutputLayer:"<<endl;
        for(int j=0;j<Lhid[LhidSum-1].Nsum;j++){
            for(int k=0;k<Lout.Nsum;k++){
                cout<<setw(10)<<"  W("<<j<<'-'<<k<<")"<<Wsums[LhidSum][j][k]<<';';
            }
            cout<<endl;
        }
        cout<<"  Bias:"<<Lout.b<<endl;
    }
    
    void PrintConfidenceDGR(){
        
        cout<<"MLP function result's printing:"<<endl<<endl;
        
        for(int i=0;i<Lout.Nsum;i++){
            cout<<"〔O-Neuron#"<<i<<"〕:[";
            for(int j=1;j<=int(*(Lout.Ldata[i].y)*100);j++){cout<<"▊";}
            cout<<"]   ( "<<int(*(Lout.Ldata[i].y)*100)<<"% )\n";
        }
        
        cout<<"\n\n";
        
    }
    
    void FeedForward(){
        //###输入层 → 隐藏层传播###
        if (Wsums == nullptr) {
            cout << "Error: Weights not initialized!" << endl;
            return;
        }
        
        double Sumup;
        
        /*
        cout<<"InputLayer -- HiddenLayer[0]:"<<endl;
        */
        
        for(int j=0;j<Lhid[0].Nsum;j++){
            Sumup=0.0;
            
            /*
            cout<<"----Lhid[0].Ldata["<<j<<"].y = ☌(";
            */
            
            for(int k=0;k<Lin.Nsum;k++){
                Sumup+= *(Lin.Ldata[k].x) * Wsums[0][k][j];
                
                /*
                cout<< *(Lin.Ldata[k].x)<<" * "<<Wsums[0][k][j]<<" + ";
                */
                
            }
            Sumup+=Lhid[0].b;
            *(Lhid[0].Ldata[j].y)=Lhid[0].actfunction(Sumup);
            
            /*
            cout<<Lhid[0].b<<")\n----                   = "<<(Lhid[0].actfunction==ActFunc_ReLU ? "ReLU" : "Sigmoid")<<"("<<Sumup<<")\n----                   = "<<*(Lhid[0].Ldata[j].y)<<endl;
            */
            
        }
        
        for(int i=1;i<LhidSum;i++){
            
            /*
            cout<<endl<<"HiddenLayer["<<i-1<<"] -- HiddenLayer["<<i<<"]:"<<endl;
            */
            
            for(int j=0;j<Lhid[i].Nsum;j++){
                Sumup=0.0;
                
                /*
                cout<<"----Lhid["<<i<<"].Ldata["<<j<<"].y = ☌(";
                */
                
                for(int k=0;k<Lhid[i-1].Nsum;k++){
                    Sumup+= *(Lhid[i-1].Ldata[k].y) * Wsums[i][k][j];
                    
                    /*
                    cout<< *(Lhid[i-1].Ldata[k].y)<<" * "<<Wsums[i][k][j]<<" + ";
                    */
                    
                }
                Sumup+=Lhid[i].b;
                *(Lhid[i].Ldata[j].y)=Lhid[i].actfunction(Sumup);
                
                /*
                cout<<Lhid[i].b<<")\n----                   = "<<(Lhid[i].actfunction==ActFunc_ReLU ? "ReLU" : "Sigmoid")<<"("<<Sumup<<")\n----                   = "<<*(Lhid[i].Ldata[j].y)<<endl;
                */
                
            }
        }
        //###隐藏层n → 输出层传播###
        
        /*
        cout<<endl<<"HiddenLayer["<<LhidSum<<"] -- OutputLayer:"<<endl;
        */
        
        for(int j=0;j<Lout.Nsum;j++){
            Sumup=0.0;
            
            /*
            cout<<"----Lout.Ldata["<<j<<"].y = ☌(";
            */
            
            for(int k=0;k<Lhid[LhidSum-1].Nsum;k++){
                Sumup+= *(Lhid[LhidSum-1].Ldata[k].y) * Wsums[LhidSum][k][j];
                
                /*
                cout<< *(Lhid[LhidSum-1].Ldata[k].y)<<" * "<<Wsums[LhidSum][k][j]<<" + ";
                */
            }
            Sumup+=Lout.b;
            *(Lout.Ldata[j].y)=Lhid[LhidSum-1].actfunction(Sumup);
            
            /*
            cout<<Lout.b<<")\n----                = "<<(Lout.actfunction==ActFunc_ReLU ? "ReLU" : "Sigmoid")<<"("<<Sumup<<")\n----                = "<<setprecision(8)<<*(Lhid[LhidSum-1].Ldata[j].y)<<endl;
            */
            
        }
        
        /*
        cout<<"\n\n";
        */
        
        PrintConfidenceDGR();
    }
    
    // 添加析构函数释放内存
    ~Net() {
        if (LhidSum > 0) {
             delete[] Lhid;
             Lhid = nullptr;  // 避免悬空指针
        }
        // 释放Wsums内存
        if (Wsums != nullptr) {
            for (int i = 0; i < LhidSum + 1; i++) {
                if (Wsums[i] != nullptr) {
                    int rows = (i == 0) ? Lin.Nsum : (i < LhidSum) ? Lhid[i - 1].Nsum : Lhid[LhidSum - 1].Nsum;
                    for (int j = 0; j < rows; j++) {
                        delete[] Wsums[i][j];
                    }
                    delete[] Wsums[i];
                }
            }
            delete[] Wsums;
        }
    }
};
int main()
{
    //初始化
    srand(time(NULL));
    FloatValue_len=11;
    
    int LhidN[1]={16};
    double LhidB[1]={0.005};
    ActFunc acts[1]={ActFunc_ReLU};
    
    Net FNN;
    /*一个
    
      .输入层：4个神经元
      .2层隐藏层：
        。隐藏层0：4个神经元
        。隐藏层1：4个神经元
      .输出层：4个神经元
      
    的神经网络
    */
    
    //###定义28*28像素的手写数字2灰度数组###
    double ImageDataInput[784]={
    //   00 |01 |02 |03 |04 |05 |06 |07 |08 |09 |10 |11 |12 |13 |14 |15 |16 |17 |18 |19 |20 |21 |22 |23 |24 |25 |26 |27
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
        //0
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
        //1
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
        //2
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
        //3
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.3,0.4,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.3,0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
        //4
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.3,0.5,0.8,1.0,1.0,1.0,1.0,1.0,1.0,0.7,0.5,0.2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
        //5
        0.0,0.0,0.0,0.0,0.0,0.0,0.3,0.5,0.8,1.0,1.0,1.0,0.8,1.0,1.0,1.0,1.0,1.0,0.4,0.2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
        //6
        0.0,0.0,0.0,0.0,0.0,0.0,0.3,0.5,1.0,0.6,0.6,0.5,0.4,0.5,0.6,0.7,1.0,1.0,0.5,0.3,0.2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
        //7
        0.0,0.0,0.0,0.0,0.0,0.0,0.2,0.5,0.9,0.4,0.2,0.1,0.1,0.2,0.3,0.5,0.6,1.0,1.0,1.0,0.5,0.2,0.0,0.0,0.0,0.0,0.0,0.0,
        //8
        0.0,0.0,0.0,0.0,0.0,0.0,0.2,0.6,0.5,0.3,0.0,0.0,0.0,0.0,0.0,0.2,0.4,0.6,1.0,1.0,0.5,0.2,0.1,0.0,0.0,0.0,0.0,0.0,
        //9
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.2,0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.3,0.5,1.0,1.0,1.0,0.4,0.1,0.0,0.0,0.0,0.0,0.0,
        //10
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.3,0.5,1.0,1.0,0.6,0.3,0.0,0.0,0.0,0.0,0.0,
        //11
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.2,0.4,0.6,1.0,1.0,0.5,0.2,0.0,0.0,0.0,0.0,0.0,
        //12
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.3,0.5,1.0,1.0,1.0,0.5,0.2,0.0,0.0,0.0,0.0,0.0,
        //13
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.3,0.5,1.0,1.0,1.0,0.5,0.2,0.0,0.0,0.0,0.0,0.0,
        //14
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.3,0.5,1.0,1.0,1.0,0.6,0.3,0.0,0.0,0.0,0.0,0.0,0.0,
        //15
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.2,0.4,0.6,1.0,1.0,0.7,0.4,0.2,0.0,0.0,0.0,0.0,0.0,0.0,
        //16
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.3,0.5,1.0,1.0,1.0,0.5,0.3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
        //17
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.2,0.4,0.6,1.0,1.0,1.0,0.5,0.2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
        //18
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.4,0.6,1.0,1.0,1.0,1.0,0.5,0.2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
        //19
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.5,0.6,0.7,1.0,1.0,1.0,1.0,0.5,0.2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
        //20
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.2,0.4,0.6,1.0,1.0,1.0,1.0,0.5,0.2,0.0,0.0,0.0,0.2,0.2,0.3,0.0,0.0,0.0,
        //21
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.3,0.3,0.2,0.4,0.0,1.0,1.0,1.0,1.0,0.6,0.2,0.1,0.3,0.3,0.5,0.5,0.9,0.5,0.2,0.0,0.0,
        //22
        0.0,0.0,0.0,0.0,0.0,0.0,0.3,0.5,0.5,0.5,0.5,1.0,1.0,1.0,0.8,0.6,0.5,0.6,0.9,0.8,0.9,1.0,1.0,1.0,0.6,0.3,0.0,0.0,
        //23
        0.0,0.0,0.0,0.0,0.0,0.3,0.5,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.4,0.2,0.0,0.0,
        //24
        0.0,0.0,0.0,0.0,0.0,0.3,0.5,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.9,0.9,0.8,0.8,0.8,0.8,0.8,0.5,0.3,0.3,0.1,0.0,0.0,0.0,
        //25
        0.0,0.0,0.0,0.0,0.0,0.3,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.4,0.3,0.2,0.2,0.2,0.1,0.1,0.1,0.0,0.0,0.0,0.0,0.0,0.0,
        //26
        0.0,0.0,0.0,0.0,0.0,0.0,0.3,0.3,0.3,0.3,0.3,0.2,0.1,0.1,0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
        //27
    };//,expectedOutput[10]={0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
    
    FNN.__init__(28*28,0.01,ActFunc_ReLU,1,LhidN,LhidB,acts,10,0.001,ActFunc_Sigmoid);
    FNN.ConnectWbyRandomV();  // 初始化权重矩阵
    
    /*
    FNN.EchoNetInfo();
    */
    
    /*cout<<endl<<endl;*/
    
    FNN.Lin.TypeLayerV(ImageDataInput,784);
    
    /*
    *(FNN.Lin.Ldata[0].x)=0.25;
    *(FNN.Lin.Ldata[1].x)=0.125;
    *(FNN.Lin.Ldata[2].x)=0.0625;
    *(FNN.Lin.Ldata[3].x)=0.03125;
    */
    
    FNN.FeedForward();
    
    return 0;
}
    