#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <vector>
#include <functional>
#include <fstream>
#include "mpi.h"
#include <algorithm>
 
typedef float Scalar;
typedef Eigen::MatrixXf Matrix;
typedef Eigen::RowVectorXf RowVector;
typedef Eigen::VectorXf ColVector;
 
Scalar activationFunction(Scalar x)
{
    return tanhf(x);
    //return std::max(0.0f,x);
  // return 1.0/exp(-(double)x);
}
 
Scalar grad(Scalar x)
{
    return 1 - tanhf(x)*tanhf(x);
    //if(x > 0)
      //return 1.0f;
    //return 0.0f;
   // return activationFunction(x)*(1.0-activationFunction(x));
}
// you can use your own code here!


class NeuralNetwork {
public:
    // constructor
    NeuralNetwork(std::vector<uint> topology, bool host = false,Scalar learningRate = Scalar(0.005));
    // function for forward propagation of data
    void propagateForward(RowVector& input);
    // function for backward propagation of errors made by neurons
    void propagateBackward(RowVector& output);
    // function to calculate errors made by neurons in each layer
    void calcErrors(RowVector& output);
    // function to update the weights of connections
    void updateWeights();
    // function to train the neural network give an array of data points
    Scalar train(int epoch,int batch,std::ifstream& fin,Scalar lr = 0.002);
    void config(std::string configfile);
    void get_train_set(std::istream&fin ,int batch_size,int scale);
    void save(std::string savefile);
    int test(std::istream& fin){
        int count = 0;
        int magic,total;
        int rows,cols;
        RowVector::Index index;
        total = 10000;
        rows = 28;
        for (int j = 0; j < total ; j++) {
            get_train_set(fin, 1 , rows);
            propagateForward(*(input_data[0]));
            output_layer->maxCoeff(&index);
            //std::cerr <<"answer =" << index << std::endl;
            if(output_data[0]->coeff(index)>0.5){
                count ++;
            }
        }
        return count;
    }
    std::vector<Matrix*> weights; // the connection weights
    int hits = 0;
    void train_iters(int iters,std::ifstream& fin){
        int rows = 28;
        hits = 0;
        RowVector::Index index;
        for(int j=0;j<iters;j++){
            //std::cerr << " start to get " << j*batch <<","<< std::endl;
            get_train_set(fin,1,rows);
            //std::cerr << "success get" << std::endl;
            for(int k=0;k < 1;k++){
                propagateForward(*(input_data[0]));
                output_layer->maxCoeff(&index);
                if(output_data[0]->coeff(index)>0.5){
                    hits++;
                }
                propagateBackward(*(output_data[0]));
               // std:: cerr << "train success" << std::endl;
            }
            //std::cerr << "epoch " << i << ", batch " << j << "finish!" << std::endl;
            if(learningRate > 0.0005)
                learningRate *= 0.95;
        }
    }
    Scalar learningRate;
   private:
    std::vector<RowVector*> input_data;
    std::vector<RowVector*> output_data;
    RowVector* output_layer = nullptr;
   protected:
    std::vector<RowVector*> neuronLayers; //layers of out network
    std::vector<RowVector*> cacheLayers; //unactivated (activation fn not yet applied) values of layers
    std::vector<RowVector*> deltas; // error contribution of each neurons
   
    std::vector<uint> topology;// network topology
    /**
     * say topology = {4,2,3} means the first layer has 4 neurons,
     * second layer has 2,and so on. these are all hidden layers.
     * in other words, input layer and output layer will be determined
     * by specific classify task.
     * But in the vector topology, for easy usage, we let topology[0] be input layer
     * and output layer for topology[-1](slightly misuse of python syntax).
     * In that way, W[i] actually presents weight matrix for layer i + 1.
     * */
};

int kik = 0;
void NeuralNetwork::get_train_set(std::istream&fin , int batch_size,int scale){
    //say scale*scale size image
    if(input_data.size()==0){
        for(int i=0;i<batch_size;i++){
            input_data.push_back(new RowVector(scale*scale));
            output_data.push_back(new RowVector(10));
        }
    }
    int pixel;
    int l;
    char comma;
    for(int i=0;i<batch_size;i++){
        //std::cerr << " what error " << i << std::endl;
        fin >> l >> comma;
        for(int j=0;j<scale*scale - 1;j++){
            fin >> pixel >> comma;
            input_data[i]->coeffRef(j) = (Scalar)pixel;
        }
        fin >> pixel;
        input_data[i]->coeffRef(scale*scale - 1) = (Scalar)pixel;
        if(kik)
            std::cerr << "target =" << l << std::endl;
        output_data[i]->setZero(10);
        output_data[i]->coeffRef(l) = 1.0f;
    }
}

void NeuralNetwork::save(std::string savefile){
    std::ofstream s(savefile);
    s << topology.size() << std::endl;
    for(auto x:topology){
        s << x << " ";
    }
    s << std::endl;
    for(int i=0;i<weights.size();i++){
        s << weights[i]->rows() << " " << weights[i]->cols() << std::endl;
        s << *(weights[i]) << std::endl;
    }
    std::cerr << "stored" << std::endl;
    s.close();
}

void NeuralNetwork::config(std::string s){
    std::ifstream in(s);
    int n;
    in >> n;
    int dim;
    for(int i=0;i<n;i++){
        in >> dim;
        topology.push_back(dim);
    }
    int row,col;
    Scalar data;
    for(int i=0;i<n;i++){
        in >> row >> col;
        weights.push_back(new Matrix(row,col));
        for(int p = 0;p<row;p++){
            for(int q = 0;q < col;q++){
                in >> data;
                weights[i]->coeffRef(p,q) = dim;
            }
        }
    }
    in.close();
}

NeuralNetwork::NeuralNetwork(std::vector<uint> topology, bool host, Scalar learningRate)
{
    this->topology = topology;
    this->learningRate = learningRate;
    for (uint i = 0; i < topology.size(); i++) {
        // initialize neuron layers
        if (i == topology.size() - 1){
            neuronLayers.push_back(new RowVector(topology[i]));
            cacheLayers.push_back(new RowVector(topology[i]));
            deltas.push_back(new RowVector(topology[i]));
        }
        else{
            neuronLayers.push_back(new RowVector(topology[i] + 1));
            cacheLayers.push_back(new RowVector(topology[i]+1));
            deltas.push_back(new RowVector(topology[i]+1));
        }
 
        // initialize cache and delta vectors
        //cacheLayers.push_back(new RowVector(neuronLayers.size()));
        //deltas.push_back(new RowVector(neuronLayers.size()));
 
        if (i != topology.size() - 1) {
            neuronLayers.back()->coeffRef(topology[i]) = 1.0;
            cacheLayers.back()->coeffRef(topology[i]) = 1.0;
        }
 
        // initialize weights matrix
        if (i > 0) {
            if (i != topology.size() - 1) {
                weights.push_back(new Matrix(topology[i - 1] + 1, topology[i] + 1));
                if(host){
                    //如果是主机(服务器),则初始化.否则不初始化.
                    weights.back()->setRandom();
                    weights.back()->col(topology[i]).setZero();
                    weights.back()->coeffRef(topology[i - 1], topology[i]) = 1.0;
                }
            }
            else {
                weights.push_back(new Matrix(topology[i - 1] + 1, topology[i]));
                if(host)
                    weights.back()->setRandom();
            }
        }
    }
};

void NeuralNetwork::propagateForward(RowVector& input)
{
    neuronLayers.front()->head(neuronLayers.front()->size() - 1) = input;
    // last one is bias, so ignore it.
 
    for (uint i = 1; i < topology.size(); i++) {
        
        //cache layer store values unactivate.
        //but cache layer generated by previous layer's output
        //btw, here is where bias is used and updated.
        (*cacheLayers[i]) = (*neuronLayers[i - 1]) * (*weights[i - 1]);

        //apply activate,but ignore bias.
        neuronLayers[i]->head(topology[i]) = cacheLayers[i]->head(topology[i]).unaryExpr(std::ref(activationFunction));

        //forward bias.
        neuronLayers[i]->coeffRef(topology[i]) = cacheLayers[i]->coeff(topology[i]);
    }
    if(output_layer == nullptr)
        output_layer = new RowVector(10);
    auto M = neuronLayers.back()->maxCoeff();
    *output_layer = neuronLayers.back()->unaryExpr([M](const Scalar& elem){return (Scalar)exp(elem-M);});
    auto sum = output_layer->sum();
    *output_layer = output_layer->unaryExpr([sum](const Scalar& elem){return elem/sum;});
}

void NeuralNetwork::calcErrors(RowVector& output)
{
    // calculate the errors made by neurons of last layer
   // (*deltas.back()) = output - (*neuronLayers.back());
  // (*deltas.back()) = output - *output_layer;
 
    //backward calculation, so we do transpose
   // for (uint i = topology.size() - 2; i > 0; i--) {
     //   (*deltas[i]) = (*deltas[i + 1]) * (weights[i]->transpose());
    //}
    (*deltas.back()) = output - *output_layer;
 
    //backward calculation, so we do transpose
    for (uint i = topology.size() - 2; i > 0; i--) {
        (*deltas[i]) = (deltas[i + 1]->cwiseProduct(cacheLayers[i+1]->unaryExpr(
            [](const Scalar& elem){return grad(elem);}
        ))) * (weights[i]->transpose());
    }
}

void NeuralNetwork::updateWeights()
{
    // topology.size()-1 = weights.size()
    for (uint i = 0; i < topology.size() - 1; i++){
        //if not last layer --- output layer
        //
        if (i != topology.size() - 2) {
            for (uint c = 0; c < weights[i]->cols() - 1; c++) {
                //last one is bias.
                auto co = learningRate * deltas[i + 1]->coeffRef(c) * 
                    grad(cacheLayers[i + 1]->coeffRef(c));
                weights[i]->col(c) += neuronLayers[i]->transpose().unaryExpr([co](const Scalar& elem){ return co*elem;});
                //for (uint r = 0; r < weights[i]->rows(); r++) {
                  //  weights[i]->coeffRef(r, c) += 
                    //neuronLayers[i]->coeffRef(r);
                    //update weight[i] but use layer i+1's cache data
                    //and layer i's output.
                    //reason explained above. weight[i] actually is layer i+1's matrix
                //}
            }
        }
        else {
            for (uint c = 0; c < weights[i]->cols(); c++) {
                auto co = learningRate * deltas[i + 1]->coeffRef(c) * 
                    grad(cacheLayers[i + 1]->coeffRef(c));
                weights[i]->col(c) += 
                    neuronLayers[i]->transpose().unaryExpr([co](const Scalar& elem)
                    { return co*elem;});
               /* for (uint r = 0; r < weights[i]->rows(); r++) {
                    weights[i]->coeffRef(r, c) += 
                    learningRate * deltas[i + 1]->coeffRef(c) *
                    grad(cacheLayers[i + 1]->coeffRef(c)) * 
                    neuronLayers[i]->coeffRef(r);
                }*/
            }
        }
    }
}

void NeuralNetwork::propagateBackward(RowVector& output)
{
    calcErrors(output);
    updateWeights();
}

Scalar NeuralNetwork::train(int epoch,int batch,std::ifstream& fin,Scalar lr)
{
    int magic,total;
    int rows,cols;
    char comma;
    learningRate = lr;
    for(int i=0;i<epoch;i++){
        total = 10000;
        rows = 28;
        for(int j=0;j<total/batch;j++){
            //std::cerr << " start to get " << j*batch <<","<< std::endl;
            get_train_set(fin,batch,rows);
            //std::cerr << "success get" << std::endl;
            for(int k=0;k < batch;k++){
                propagateForward(*(input_data[k]));
                propagateBackward(*(output_data[k]));
               // std:: cerr << "train success" << std::endl;
            }
            //std::cerr << "epoch " << i << ", batch " << j << "finish!" << std::endl;
            if(learningRate > 0.0005)
                learningRate *= 0.95;
        }
        fin.clear();
        fin.seekg(0);//back to beginning
    }
    return learningRate;
}

class Weights{
   public:
    Weights(std::vector<uint> topology){
        for (uint i = 0; i < topology.size(); i++) {
    
           //纯纯的矩阵传递人
            if (i > 0) {
                if (i != topology.size() - 1) {
                    weights.push_back(new Matrix(topology[i - 1] + 1, topology[i] + 1));
                }
                else {
                    weights.push_back(new Matrix(topology[i - 1] + 1, topology[i]));
                }
            }
        }
    }
    std::vector<Matrix*> weights; // the connection weights
};

//the variant of config and store is recv and send for MPI model.
//we send(store) weights to centeral server, and server recv(config,load) them.

void Recv_Weight_MPI(NeuralNetwork& N,Weights& temp,int src,int tag,int order){
    int layers = N.weights.size();
    MPI_Status Delc;
    if(order == 0){
        for(int i=0;i<layers;i++){
            MPI_Recv(N.weights[i]->data(),N.weights[i]->size(),MPI_FLOAT,src,tag+i,MPI_COMM_WORLD,&Delc);
        }
    }
    else{
        for(int i=0;i<layers;i++){
            MPI_Recv(temp.weights[i]->data(),N.weights[i]->size(),MPI_FLOAT,src,tag+i,MPI_COMM_WORLD,&Delc);
            //average
        }
        for(int i=0;i<layers;i++){
            *N.weights[i] = N.weights[i]->unaryExpr([order](const float& elem)
            {return elem*(float)(order)/(float)(order+1);}) + temp.weights[i]->unaryExpr([order](const float& elem)
            {return (Scalar)(elem*1.0/(float)(order+1));});
        }
    }
}

void Send_Weight_MPI(NeuralNetwork& N,int dest,int tag){
    int layers = N.weights.size();
    for(int i=0;i<layers;i++){
        MPI_Send(N.weights[i]->data(),N.weights[i]->size(),MPI_FLOAT,dest,tag+i,MPI_COMM_WORLD);
    }
}
#include "unistd.h"
#include <sys/types.h>
#include <chrono>
int main(void){

    MPI_Init(NULL, NULL);

    // 通过调用以下方法来得到所有可以工作的进程数量
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // 得到当前进程的秩
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int* result;
    int hits = 0;
    int epoch = 30;
    int batch = 10000;
    int workers = world_size - 1;
    pid_t pid = getpid();
    using namespace std;
    vector<Matrix*> best_half;
    vector<pair<int,int>> get_best(workers);
    if(rank == 0){
        cerr << pid << endl;
        auto start = chrono::steady_clock::now();
        typedef std::chrono::high_resolution_clock Time;
    typedef std::chrono::milliseconds ms;
    typedef std::chrono::duration<float> fsec;
        auto t0 = Time::now();
        result = (int*)malloc(sizeof(int)*world_size);
        NeuralNetwork NN({28*28,1000,1000,10},true,0.002f);//主机
        Weights TEMP({28*28,1000,1000,10});
        for(int uu = 0;uu<epoch;uu++){
            //cerr << uu <<endl;
            for(int i=0;i<NN.weights.size();i++){
                auto ptr = NN.weights[i];
                MPI_Bcast(ptr->data(),ptr->size(),MPI_FLOAT,0,MPI_COMM_WORLD);
            }
            
            //cerr << "say i am ok" << endl;
            MPI_Gather(&hits,1,MPI_INT,result,1,MPI_INT,0,MPI_COMM_WORLD);
            for(int i=0;i<workers;i++){
                get_best[i] = {result[i+1],i+1};
            }
            std::sort(get_best.begin(),get_best.end(),greater<pair<int,int>>());
            for(int i=0;i<workers;i++){
                int dest = get_best[i].second;
                if(i<workers/2)
                    result[dest] = 1;//猪猡，干活喽
                else
                    result[dest] = 0;
            }
            //cerr << "assign" << endl;
            MPI_Scatter(result,1,MPI_INT,&hits,1,MPI_INT,0,MPI_COMM_WORLD);
            //cerr << "task comfirm" << endl;
            for(int i=0;i<workers/2;i++){
                int dest = get_best[i].second;
                Recv_Weight_MPI(NN,TEMP,dest,dest*2,i);
            }
            //权值矩阵已在Recv中更新.
        }
        auto t1 = Time::now();
        fsec fs = t1 - t0;
        ms d = std::chrono::duration_cast<ms>(fs);
        std::cerr << "training time = " << d.count() << std::endl;
        std::cerr << "go to test" << std::endl;
        //训练完毕.
        NN.save("para-model.nn");
        std::cerr << "ready to test " << std::endl;
        std::ifstream test("./data/mnist_test.csv");
        int result = NN.test(test);
        std::cerr << "result = " << result << std::endl;
    }
    else{
        NeuralNetwork NN({28*28,1000,1000,10},false,0.002f);
        std::string path = "./data/train" + std::to_string(rank-1) + ".csv";
        std::ifstream train(path);
        for(int uu=0;uu<epoch;uu++){
            for(int i=0;i<NN.weights.size();i++){
                auto ptr = NN.weights[i];
                MPI_Bcast(ptr->data(),ptr->size(),MPI_FLOAT,0,MPI_COMM_WORLD);
            }
            //cerr << "rank" << rank << "start train" << endl;
            NN.train_iters(batch,train);
            //std::cerr << "rank" << rank << " train finish" << std::endl;
            hits = NN.hits;
            MPI_Gather(&hits,1,MPI_INT,NULL,0,MPI_INT,0,MPI_COMM_WORLD);
            MPI_Scatter(NULL,0,MPI_INT,&hits,1,MPI_INT,0,MPI_COMM_WORLD);
            if(hits == 1){
                Send_Weight_MPI(NN,0,rank*2);
            }
            if(train.eof()){
                train.clear();train.seekg(0);
            }
        }
        //auto r = NN.train(30,20,train,0.002);
        //std::ifstream train1("./data/train5.csv");
        //NN.train(30,20,train1,r);
    }
    MPI_Finalize();
    return 0;
}