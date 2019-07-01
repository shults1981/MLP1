/*
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/


/*
|*****************************************************************
|*****************************************************************
* Project                        :  <Multilayer perceptron>
* Programm name                  :  MLP1.c
* Author                         :  Shults1981
* Data create                    :  01/12/2012
* Purpose                        :  conceived as a test implementation
*                                   of a multilayer perceptron for microcontrollers!
                                    But ... later implemented as a console application
                                    for a general-purpose machine.
|****************************************************************
|****************************************************************
*/


#define DEBUG



#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<string.h>
#include<ctype.h>
//#include<setjmp.h>






typedef struct ML_perceptron
{
    int X;           // size of input vector(размерность входного вектора)
    int Y;           // size of output vector(размерность выходного вектора
    int L;           // numbers of layers (кол-во слоёв)
    int *J;          // number of neurons in the test layers except the last(кол-во нейронов в скпытых слоях кроме последнего)
    double *Xin;     // input vector (входной вектор)
    double *Yout;    // output vector (выходной вектор)
    double ***x;     // input vector for neuron (входные сигналы нейрона)
    double ***w;     // neuron weight coefficients (весовые коэфициенты нейрона)
    double **NET;    // weighted neuron sum (взвешенная сумма нейрона)
    double **Out;    // neuron output (выход  нейронов)
    double **O;      // threshold level of the neuron (offset) (пороговый уровень нейрона(смещение))
    int FofA;        // activation function number (номер функции активации)

    //auxiliary variables to normalize input and output(вспомогательные переменные для нормализации входных и выходных данных)
    double deltaX;
    double deltaY;
    double alphaX;
    double alphaY;
} MLP;


struct  trVec    // structure of the training vector (структура обучающего вектора)
{
    double *Xin;  // input vector (входной вектор)
    double *Yout; // output vector (выходной вектор)
};


//-------------------------------PROTOTYPES OF FUNCTIONS/ ПРОТОТИПЫ ФУНКЦИЙ----------------------------------------------
void main_menu (MLP*);                  //the menu for choosing from which structure to work new / saved to a file
                                        //(меню выбора из какой структуры работать новая/сохраненная в файл)

int op_mode_menu(void);                 // Work / Training / Saving selection menu (меню выбора Работа/Обучение/Сохранение)

int tr_mode_menu(void);                 // selection menu(меню выбора): -create data training file(создать файл с обуч данными)
                                        //                              -view training data(посмотреть обучающие данные)
                                        //                              -start learning(начать обучение)

void initialization_MLP_new ( MLP *);   // initialization of the new MLP with the input of the main parameters
                                        //инициализация нового МСП с вводом основных параметров

void initialization_MLP_File ( MLP *);  // initialization of saved SMEs from file(инициализация сохраненного МСП с из файла)

int  work_MLP(MLP*);                    // MLP operation in the calculation mode(работа МСП в режиме расчета)

void basic_calc_MLP(MLP *, double*);    // basic calculation of SMEs (calculation of input output) (основной расчет МСП(вычисление выхода по входу))

void training_MLP(MLP*);                // training MLP(обучения МСП)

void save_MLP_File ( MLP *);            // saved MLP to file (сохранение МСП в файл)

void delete_MLP (MLP *);                // removing a MLP from memoryудаление (многослойного пересептрона из памяти)

void creat_train_data_file (MLP*);      // creating a file with a training vector(создание файла с обучающим вектором)

void output_train_data(MLP*);           // view training data(просмотр обучающих данных)

int  num_control (char *,int, int, int, int); // check function on( ф-ия проверки на):
                                             //  -numder (число(целое/вещ.определяется перем DorI))
                                             //  -belonging to the interval(принадлежность интервалу (min;max)
                                             //  -Belonging to negative values(принадлежность к отрицательным значениям SorUS (min задавать <0))

double (*funk_of_act) (double);      // pointer to the activation function(указатель на функцию активации)

double (*Der_funk_of_act)(double);   // pointer to derivative function activation (указатель на производную ф-ии активации)

double sigmoid (double);             // sigmoid activation function (logistic) (функция активации сигмоид(логистическая))

double gip_tan (double);             // hyperbolic tangent activation function (функция активации гиперболический тангенс)

double Der_sigmoid (double);         // derivative of activation function - sigmoid (производная функции активации сигмоид)

double Der_gip_tan (double);         // hyperbolic tangent activation function derivative производная функции активации гиперболический тангенс

void rating01 (struct trVec *tV, int M, int X, int Y, MLP *);// vector normalization to the interval [0; 1] (нормализация вектора к интервалу [0;1])

void rating11 (struct trVec *tV, int M, int X, int Y, MLP *); // vector normalization to the interval [0; 1] (нормализация вектора к интервалу [-1;1])

double RMS_error (MLP*,struct trVec *, int,int,int);// mean square error (среднеквадратическая ошибка)

void mixing_trVec(struct trVec *,int, int ,int); // mixing the training vector (перемешивание обучающего вектора)



//---------------------------------------------------------------------------------


//*********************************************************************************
//********************************** MAIN *********************************
int
main()
{

    int i;
    int j;
    int l;
    int iMenu1;
    int iMenu2;
    MLP *P1;


    P1=(MLP*)malloc(sizeof(MLP));
    P1->X=0;
    P1->Y=0;
    P1->L=0;

//------------------------------------------------KERNEL / ЯДРО---------------------------------------


 for(;;)
 {

    main_menu(P1); //selection of the method of initialization of MLP  (выбор способа инициализации МСП)


#ifdef DEBUG //------------------------------debug information/отладочная информация-------------------
    printf ("************ initialize parameters of MLP **********************\n");
    printf ("\n");
    printf("dimension of INPUT  vector:-   %d\n",P1->X);
    printf("dimtnsion of OUTPUT vector:-   %d\n",P1->Y);
    printf("number of layers:-             %d\n",P1->L);
    printf("input layer:-                  %d\n",P1->J[0]);
    for (l=1;l<(P1->L-1);l++)
        printf("neurons in layer:%d-            %d\n",l,P1->J[l]);
    printf("neurons in output layer:-      %d\n",P1->J[P1->L-1]);

    if (P1->FofA==1)
        printf("activation function:- logistic function\n");
    else
        printf("activation function:- hyperbolic tangent\n");

    printf("deltaX=                         %f\n",P1->deltaX);
    printf("deltaY=                         %f\n",P1->deltaY);
    printf("alphaX=                         %f\n",P1->alphaX);
    printf("alphaY=                         %f\n",P1->alphaY);
    printf("****************************************************************\n");

#endif  //--------------------------------------------------------------------------------------


    while ((iMenu1=op_mode_menu())!=4) //large selection cycle TRAINING / WORK / PRESERVATION (большой цикл выбора ОБУЧЕНИЕ/РАБОТА/СОХРАНЕНИЕ)
    {

        switch (iMenu1)
        {
        case 1:
        {
            while((iMenu2=tr_mode_menu())!=4)// small cycle selection in the training menu (малый цикл выбора в меню обучения)
                switch (iMenu2)
                {
                case 1:creat_train_data_file(P1);break;
                case 2:output_train_data(P1);break;
                case 3:training_MLP(P1);break;
                case 4:break;
                }
        }
            break;



        case 2:
        {
            while (work_MLP(P1)!=0)// cycle work mode work (цикл работа в режиме работа)
            {

#ifdef DEBUG //-----------------------------------debug information/отладочная информация---------------------
                printf("\n");
                printf("                             internal data of MLP\n");
                //--------input vector/вектор входов---------------------------------
                printf("inputs of neurons\n");
                for (l=1;l<P1->L;l++)
                {
                    printf("      layer %d            \n",l);
                    for (j=0;j<P1->J[l];j++)
                    {
                        for(i=0;i<P1->J[l-1];i++)
                            printf("-x%d %d %d=%f-",l,j,i,P1->x[l][j][i]);

                        printf("\n");
                    }

                }

                printf("\n");
                //--------vector of wieghts(вектор весов)---------------------------------
                printf("synapses of neurons\n");
                for (l=1;l<P1->L;l++)
                {
                    printf("         layer %d          \n",l);
                    for (j=0;j<P1->J[l];j++)
                    {
                        for(i=0;i<P1->J[l-1];i++)
                            printf("-w%d %d %d=%f-",l,j,i,P1->w[l][j][i]);
                        printf("\n");
                    }
                }

                printf("\n");
                //---------threshold vector (вектор пороговых коэфициентов)-----------
                printf("bias coefficients\n");
                for (l=1;l<P1->L;l++)
                {
                    printf("         layer %d          \n",l);
                    for(j=0;j<P1->J[l];j++)
                        printf("-O%d %d=%f",l,j,P1->O[l][j]);
                    printf("\n");
                }
                printf("******************************************************\n");
#endif  //-----------------------------------------------------------------------------------


            }
            break;
        }
        case 3: save_MLP_File ( P1);break;
        case 4: break;
        }

    }

 }
//-----------------------------------KERNEL END / КОНЕЦ ЯДРА------------------------------------------------------

    return 0;
}
//*********************************** END OF  MAIN ************************************************
//***********************************************************************************************








//------------------------------------------------------------------------------------
//---------------------------functions and procedures (функции и процедуры)-----------
//------------------------------------------------------------------------------------


void main_menu(MLP *P1)
{
    int flag;
    char *temp;

    temp=(char*)malloc(3*sizeof(char));
    system("cls");//очистка для Windous
    //system("clrscr");//очистка для Linux
    printf("Multilayer perceptron v0.0.1\n");
    printf("\n");
    printf("        MAIN MANU\n");
    printf("1: Initialize new structure\n");
    printf("2: Load structure from file\n");
    printf("3: Exit\n");
    flag=0;
    while(!flag)
    {
        fgets(temp,3,stdin);
        flag=num_control(temp,3,1,0,0);
        if (flag==0)  printf("Try again! \n");
        //rewind(stdin); //для Linux
        fflush(stdin); //для Windows

    }

    switch (atoi(temp))
    {
    case 1:
    {
        if((P1->X!=0)&&(P1->Y!=0)&&(P1->L!=0))//очищаем если создаем не новый МСП
            delete_MLP(P1);//деструктор
        initialization_MLP_new(P1);break;
    }
    case 2:
    {
        if((P1->X!=0)&&(P1->Y!=0)&&(P1->L!=0))//очищаем если создаем не новый МСП
            delete_MLP(P1);//деструктор
        initialization_MLP_File ( P1);break;
    }
    case 3:
    {
        if((P1->X!=0)&&(P1->Y!=0)&&(P1->L!=0))
            delete_MLP(P1);//деструктор
        free(P1);
        exit(1);
    }

    }

    printf("\n");

    free(temp);

}


//-------------------------------------------------------------------------------------------------------

int op_mode_menu(void)
    {

    int flag;
    char *temp;

    temp=(char*)malloc(3*sizeof(char));
    printf("\n");
    printf(" OPERATION MODE MENU\n");
    printf("1: Taining MLP\n");
    printf("2: Working with MLP\n");
    printf("3: Save MLP\n");
    printf("4: Go to main menu\n");
    flag=0;
    while(!flag)
    {
        fgets(temp,3,stdin);
        flag=num_control(temp,4,1,0,0);
        if (flag==0)  printf("Try again! \n");
        //rewind(stdin); //для Linux
        fflush(stdin); //для Windows

    }

    switch (atoi(temp))
    {
    case 1: return 1;
    case 2: return 2;
    case 3: return 3;
    case 4: return 4;
    }

    free(temp);

    return 0;

    }

//-------------------------------------------------------------------------------------------------------
int tr_mode_menu(void)
{
    int flag;
    char *temp;

    temp=(char*)malloc(3*sizeof(char));
    printf("\n");
    printf("    TRAIN MENU\n");
    printf("1: Create training data file\n");
    printf("2: Look for training data\n");
    printf("3: Training\n");
    printf("4: Go to operation mode menu\n");
    flag=0;
    while(!flag)
    {
        fgets(temp,3,stdin);
        flag=num_control(temp,4,1,0,0);
        if (flag==0)  printf("Try again! \n");
        //rewind(stdin); //для Linux
        fflush(stdin); //для Windows

    }

    switch (atoi(temp))
    {
    case 1: return 1;
    case 2: return 2;
    case 3: return 3;
    case 4: return 4;
    }

    free(temp);

    return 0;
}

//-------------------------------------------------------------------------------------------------------

void initialization_MLP_new (MLP *mlp)
     {


        int i;
        int j;
        int l;
        int flag;
        char *temp;

        temp=malloc(sizeof(char)*5);
        srand(time(NULL));

      //---------вводим размерность входного вектора с проверкой ввода
        flag=0;
        printf("Enter the dimension of input vector (between 1 and 99)\n");
        while((!flag)||(flag==-1))
        {
           fgets(temp,5,stdin);
           flag=num_control(temp,99,1,0,0);
           if ((flag==0)||(flag==-1))  printf("Enter data not correct! Try again! \n");
           else mlp->X=atoi(temp);
           //rewind(stdin); //для Linux
           fflush(stdin); //для Windows
        }

     //-----------------------------------------------------------------

     //---------вводим размерность выходного вектора с проверкой ввода


        flag=0;
        printf("Enter the dimension of output vector (between 1 and 99)\n");
        while((!flag)||(flag==-1))
        {
            fgets(temp,5,stdin);
            flag=num_control(temp,99,1,0,0);
            if ((flag==0)||(flag==-1))  printf("Enter data not correct! Try again! \n");
            else mlp->Y=atoi(temp);
            //rewind(stdin); //для Linux
            fflush(stdin); //для Windows
        }

        //---------------------------------------------------------------

        //---------вводим колличество скрытых слоев с проверкой ввода


        flag=0;
        printf("Enter the number of hidden layers (between 1 and 99)\n");
        while((!flag)||(flag==-1))
        {
            fgets(temp,5,stdin);
            flag=num_control(temp,99,0,0,0);
            if ((flag==0)||(flag==-1))  printf("Enter data not correct! Try again! \n");
            else mlp->L=atoi(temp)+2;
            //rewind(stdin); //для Linux
            fflush(stdin); //для Windows
        }
      //----------------------------------------------


      //---------вводим колличество нейронов в каждом слоев с проверкой ввода



        mlp->J=malloc(mlp->L*sizeof(int));
        mlp->J[0]=mlp->X; // первому слою соответствует входной вектор. так легче вести расчет
        for (l=1;l<(mlp->L-1);l++)
        {
            flag=0;
            printf("enter the number of neurons in layer %d\n",l);
            while((!flag)||(flag==-1))
            {
                fgets(temp,5,stdin);
                flag=num_control(temp,99,1,0,0);
                if ((flag==0)||(flag==-1))  printf("Enter data not correct! Try again! \n");
                else mlp->J[l]=atoi(temp);
                //rewind(stdin); //для Linux
                fflush(stdin); //для Windows
            }
        }
        mlp->J[mlp->L-1]=mlp->Y;// здесь последнем слою задается колличество нейронов
                                // равное размерности выходного векторв


        //выбираем функцию активации: сигмоид или гиперболический тангенс


        printf("enter activation function\n");
        printf("1:  logistic function \n");
        printf("2:  hyperbolic tangent \n");
        flag=0;
        while((!flag)||(flag==-1))
        {
            fgets(temp,5,stdin);
            flag=num_control(temp,2,1,0,0);
            if ((flag==0)||(flag==-1))  printf("Enter data not correct! Try again! \n");
            else mlp->FofA=atoi(temp);
            //rewind(stdin); //для Linux
            fflush(stdin); //для Windows
        }

        // ----------------инициализируем входной вектор ---------------


        mlp->Xin=malloc(mlp->X*sizeof(double));// создаем входной вектор

        for (i=0;i<mlp->X;i++)
            mlp->Xin[i]=0.0;//обнуляем входной вектор

        //--------------------------------------------------


        // ----------------инициализируем выходной вектор ---------------


        mlp->Yout=malloc(mlp->Y*sizeof(double));// создаем выходной вектор

        for (i=0;i<mlp->Y;i++)
            mlp->Yout[i]=0.0;//обнуляем выходной вектор

        //--------------------------------------------------



        //-----------инициализируем массив входов нейронов---------

        mlp->x=malloc(mlp->L*sizeof(int**));
        for (l=1;l<mlp->L;l++)
        {
            mlp->x[l]=malloc(mlp->J[l]*sizeof(int*));
            for (j=0;j<mlp->J[l];j++)
                mlp->x[l][j]=malloc((mlp->J[l-1])*sizeof(double));
        }

        for (l=1;l<mlp->L;l++)
            for(j=0;j<mlp->J[l];j++)
                for(i=0;i<mlp->J[l-1];i++)
                    mlp->x[l][j][i]=0.0;//--присваиваем 0 каждому входу каждого нейрона




        //---------------------инициализируем массив весовых коэфициентов случайными числами---
        mlp->w=malloc(mlp->L*sizeof(int**));
        for (l=1;l<mlp->L;l++)
        {
            mlp->w[l]=malloc(mlp->J[l]*sizeof(int*));
            for (j=0;j<mlp->J[l];j++)
                mlp->w[l][j]=malloc((mlp->J[l-1])*sizeof(double));
        }

        for (l=1;l<mlp->L;l++)
            for(j=0;j<mlp->J[l];j++)
                for(i=0;i<mlp->J[l-1];i++)
                    mlp->w[l][j][i]=((double) rand())/((double)RAND_MAX);//-присваиваем случ число
                                                                        // от 0 до 1 каждому входу


        //---------------------инициализируем массив взвешенных сумматоров нулями---

        mlp->NET=malloc(mlp->L*sizeof(int*));
        for (l=1;l<mlp->L;l++)
            mlp->NET[l]=malloc((mlp->J[l])*sizeof(double));

        for (l=1;l<mlp->L;l++)
            for(j=0;j<mlp->J[l];j++)
                mlp->NET[l][j]=0.0;


        //---------------------инициализируем массив выходов нейронов нулями---

        mlp->Out=malloc(mlp->L*sizeof(int*));
        for (l=0;l<mlp->L;l++)
            mlp->Out[l]=malloc((mlp->J[l])*sizeof(double));

        for (l=0;l<mlp->L;l++)
            for(j=0;j<mlp->J[l];j++)
                mlp->Out[l][j]=0.0;


        //---------------------инициализируем пороговых коэфициентов нейронов нулями---

        mlp->O=malloc(mlp->L*sizeof(int*));
        for (l=1;l<mlp->L;l++)
            mlp->O[l]=malloc((mlp->J[l])*sizeof(double));

        for (l=1;l<mlp->L;l++)
            for(j=0;j<mlp->J[l];j++)
                mlp->O[l][j]=0.0;

        //---------------------инициализируем функцию активации по её коду---

        switch (mlp->FofA)
        {
        case 1:
        {
            funk_of_act=sigmoid;
            Der_funk_of_act=Der_sigmoid;
        }
            break;

        case 2:
        {
            funk_of_act=gip_tan;
            Der_funk_of_act=Der_gip_tan;
        }
            break;
        }


        mlp->alphaX=0;
        mlp->alphaY=0;
        mlp->deltaX=1;
        mlp->deltaY=1;

        free(temp);// удаляем временную переменную temp
}

//----------------------------------------------------------------------------------------------------


void initialization_MLP_File ( MLP *mlp )
{

    int i;
    int j;
    int l;
    int temp1;
    double temp2;
    char *temp3;
    char *file_name;
    FILE *file;
    char marker[7]="______";

    file=NULL;
    temp3=calloc(10,sizeof(char));
    file_name=calloc(15,sizeof(char));


    while (!((file!=NULL)&&(strcmp(marker,"cfgMLP")==0)))
    {
    printf("Enter file name:");
    fgets(temp3,15,stdin);

/*    if (strcmp(temp3,"ex\n")==0) //выход из процедуры
        return;
 */
    strncpy(file_name,temp3,strlen(temp3)-1);
    //strncat(file_name,".mp",3);
    file=fopen(file_name,"rb");
    if (file==NULL)

        printf("Error! Can not open file!!!\n");
    else
    {
        fread(&marker,sizeof(marker),1,file);
        if  (strcmp(marker,"cfgMLP")!=0)
        {
            printf("it`s a not config file\n");
            fclose(file);
        }
    }
    //rewind(stdin); //для Linux
    fflush(stdin); //для Windows
    }




    fread(mlp,sizeof(MLP),1,file); //читаем из файла основную структуру МСП

   //---------вводим колличество нейронов в каждом слоев с проверкой ввода
    mlp->J=malloc(mlp->L*sizeof(int));
    mlp->J[0]=mlp->X;
    for (l=1;l<(mlp->L-1);l++)//читаем из файла кол-во нейронов по слоям
    {
        fread(&temp1,sizeof(int),1,file);
        mlp->J[l]=temp1;
    }
     mlp->J[mlp->L-1]=mlp->Y;


    // ----------------инициализируем входной вектор ---------------


    mlp->Xin=malloc(mlp->X*sizeof(double));// создаем входной вектор

    for (i=0;i<mlp->X;i++)
        mlp->Xin[i]=0.0;//обнуляем входной вектор


    // ----------------инициализируем выходной вектор ---------------

    mlp->Yout=malloc(mlp->Y*sizeof(double));// создаем входной вектор

    for (i=0;i<mlp->Y;i++)
        mlp->Yout[i]=0.0;//обнуляем выходной вектор


    //-----------инициализируем массив входов нейронов---------

    mlp->x=malloc(mlp->L*sizeof(int**));
    for (l=1;l<mlp->L;l++)
    {
        mlp->x[l]=malloc(mlp->J[l]*sizeof(int*));
        for (j=0;j<mlp->J[l];j++)
            mlp->x[l][j]=malloc((mlp->J[l-1])*sizeof(double));
    }

    for (l=1;l<mlp->L;l++)
        for(j=0;j<mlp->J[l];j++)
            for(i=0;i<mlp->J[l-1];i++)
                mlp->x[l][j][i]=0.0;//--присваиваем 0 каждому входу каждого нейрона




    //---------------------инициализируем массив весовых коэфициентов ---
    mlp->w=malloc(mlp->L*sizeof(int**));
    for (l=1;l<mlp->L;l++)
    {
        mlp->w[l]=malloc(mlp->J[l]*sizeof(int*));
        for (j=0;j<mlp->J[l];j++)
            mlp->w[l][j]=malloc((mlp->J[l-1])*sizeof(double));
    }

    for (l=1;l<mlp->L;l++)
        for(j=0;j<mlp->J[l];j++)
            for(i=0;i<mlp->J[l-1];i++)
            {
                fread(&temp2,sizeof(double),1,file);//читаем из файла значения
                mlp->w[l][j][i]=temp2;
            }

    //---------------------инициализируем массив взвешенных сумматоров нулями---

    mlp->NET=malloc(mlp->L*sizeof(int*));
    for (l=1;l<mlp->L;l++)
        mlp->NET[l]=malloc((mlp->J[l])*sizeof(double));

    for (l=1;l<mlp->L;l++)
        for(j=0;j<mlp->J[l];j++)
            mlp->NET[l][j]=0.0;


    //---------------------инициализируем массив выходов нейронов нулями---

    mlp->Out=malloc(mlp->L*sizeof(int*));
    for (l=0;l<mlp->L;l++)
        mlp->Out[l]=malloc((mlp->J[l])*sizeof(double));

    for (l=0;l<mlp->L;l++)
        for(j=0;j<mlp->J[l];j++)
            mlp->Out[l][j]=0.0;


    //---------------------инициализируем пороговых коэфициентов нейронов нулями---

    mlp->O=malloc(mlp->L*sizeof(int*));
    for (l=1;l<mlp->L;l++)
        mlp->O[l]=malloc((mlp->J[l])*sizeof(double));

    for (l=1;l<mlp->L;l++)
        for(j=0;j<mlp->J[l];j++)
        {
            fread(&temp2,sizeof(double),1,file);//читаем из файла значения
            mlp->O[l][j]=temp2;
        }

        //---------------------инициализируем функцию активации по её коду---
    switch (mlp->FofA)
    {
    case 1:
    {
        funk_of_act=sigmoid;
        Der_funk_of_act=Der_sigmoid;
    }
        break;
    case 2:
    {
        funk_of_act=gip_tan;
        Der_funk_of_act=Der_gip_tan;
    }
        break;
    }


    fclose(file);
    free(temp3);
    free(file_name);
}


//---------------------------------------------------------------------------------------------------------

void delete_MLP (MLP *mlp)
{

    int j;
    int l;

    //---------------------удаляем массив входов нейронов-------------------------------------------

    for (l=1;l<mlp->L;l++)
    {
        for (j=0;j<mlp->J[l];j++)
            free(mlp->x[l][j]);
        free(mlp->x[l]);
    }
    free(mlp->x);

    //---------------------удаляем массив весовых коэфициентов нейронов----------------------------
    for (l=1;l<mlp->L;l++)
    {
        for (j=0;j<mlp->J[l];j++)
            free(mlp->w[l][j]);
        free(mlp->w[l]);
    }
    free(mlp->w);

    //--------------------удаляем массив взвешенных сумматоров--------------------
    for (l=1;l<mlp->L;l++)
        free(mlp->NET[l]);
    free (mlp->NET);

    //--------------------удаляем массив выходов нейроно----------------------
    for (l=0;l<mlp->L;l++)
        free(mlp->Out[l]);
    free (mlp->Out);

    //--------------------удаляем массив пороговых коэфициентов----------------------
    for (l=1;l<mlp->L;l++)
        free(mlp->O[l]);
    free (mlp->O);

    //---------------------удаляем массив нейронов в слоях-------------------------------------------
    free (mlp->J);


    //---------------------удаляем входной вектор-------------------------------------------
    free (mlp->Xin);

    //---------------------удаляем выходной вектор-------------------------------------------
    free (mlp->Yout);

}


//----------------------------------------------------------------------------------------------------

void save_MLP_File ( MLP *mlp )
{
    int i,j,l;
    int tmp1;
    double tmp2;
    char *temp3;
    char *file_name;
    FILE *file;
    char marker[7]="cfgMLP";// маркер конфигурационного файла


    file=NULL;
    temp3=calloc(10,sizeof(char));
    file_name=calloc(15,sizeof(char));

    while (file==NULL)
    {
    printf("Enter file name to save configuration of MLP:");
    fgets(temp3,15,stdin);

/*    if (strcmp(temp3,"ex\n")==0) //выход из процедуры
        return;
 */
    strncpy(file_name,temp3,strlen(temp3)-1);
    //strncat(file_name,".mp",3);
    file=fopen(file_name,"wb");
    if (file==NULL)
        printf("Error! Can not open file!!!\n");
    //rewind(stdin); //для Linux
    fflush(stdin); //для Windows
    }


    fwrite(marker,sizeof(marker),1,file);//помечаем файл маркером config
    fwrite(mlp,sizeof(MLP),1,file);//сохраним осн. данные МСП


    for (l=1;l<(mlp->L-1);l++) //сохраняем кол-во нейронов по скытым слоям
    {
        tmp1=mlp->J[l];
        fwrite(&tmp1,sizeof(int),1,file);
    }

    for(l=1;l<mlp->L;l++)// сохраняем весовые коэфициенты
        for (j=0;j<mlp->J[l];j++)
            for(i=0;i<mlp->J[l-1];i++)
            {
                tmp2=mlp->w[l][j][i];
                fwrite(&tmp2,sizeof(double),1,file);
            }


    for(l=1;l<mlp->L;l++)// сохраняем пороговые коэфициенты
        for (j=0;j<mlp->J[l];j++)
        {
           tmp2=mlp->O[l][j];
           fwrite(&tmp2,sizeof(double),1,file);
        }

    if (file!=NULL)
    {
        printf("\n");
        printf("configuration file- %s was create successfull",file_name);
        printf("\n");
    }

    fclose(file);
    free(temp3);
    free(file_name);

}

//----------------------------------------------------------------------------------------------------


int work_MLP(MLP* mlp)
{

    int i;
    int j;
    int flag;
    char *temp;
    double *X;
    double *Y;

    temp=malloc(5*sizeof(char));
    X=malloc(mlp->X*sizeof(double));
    Y=malloc(mlp->Y*sizeof(double));

    //---ввод входного вектора  с консоли
    printf("operation mode - WORK(for exit enter 'ex'!)\n");
    for (i=0;i<mlp->X;i++)
    {
        flag=0;

        printf("enter X%d=",i+1);
        while(!flag)
        {
            fgets(temp,5,stdin);
            flag=num_control(temp,1000,-1000,1,1);
            if (flag==-1) break;
            if (flag==0)  printf("Enter data not correct! Try again! \n");
            else X[i]=atof(temp);
            //rewind(stdin); //для Linux
            fflush(stdin); //для Windows
        }
        if (flag==-1) break;
        printf("\n");
    }
    if (flag==-1) return 0;

    //-------нормализация ВХОДА
    if (mlp->FofA==1)
        for (i=0;i<mlp->X;i++)
            X[i]=(X[i]-mlp->alphaX)/mlp->deltaX;
    else
        for (i=0;i<mlp->X;i++)
            X[i]=2*(X[i]-mlp->alphaX)/mlp->deltaX-1;

    //-------расчет персепптрона
    basic_calc_MLP(mlp,X);

    //-------денормализация ВЫХОДА
    if (mlp->FofA==1)
        for (j=0;j<mlp->Y;j++)
            Y[j]=mlp->Yout[j]*mlp->deltaY+mlp->alphaY;
    else
        for (j=0;j<mlp->Y;j++)
            Y[j]=(mlp->Yout[j]+1)*mlp->deltaY/2+mlp->alphaY;

    //-------вывод значений выходного вектора------------------
    for (j=0;j<mlp->Y;j++)
        printf("Y%d=%f\n",j+1,Y[j]);

    free(temp);
    free(X);
    free(Y);
    return 1;

}
//---------------------------------------------------------------------------------------------------

void basic_calc_MLP(MLP * mlp, double* X)
{
    int i;
    int l;
    int j;


    for (i=0;i<mlp->X;i++)
    {
        mlp->Xin[i]=X[i];
    }

    //---передача  входного вектора в МСП

    for (l=1,j=0;j<mlp->J[l];j++)
        for (i=0;i<mlp->J[l-1];i++)
            mlp->x[l][j][i]=mlp->Xin[i];

    for (l=0,j=0;j<mlp->J[l];j++)
         mlp->Out[l][j]=mlp->Xin[j];


    //------основной расчет-------------

    for (l=1;l<mlp->L;l++)
    {
        for (j=0;j<mlp->J[l];j++)
        {
            mlp->NET[l][j]=0;
            for(i=0;i<mlp->J[l-1];i++)
            {
                mlp->NET[l][j]=mlp->NET[l][j]+(mlp->x[l][j][i]*mlp->w[l][j][i]);
            }
            mlp->Out[l][j]=(*funk_of_act)(mlp->NET[l][j]-mlp->O[l][j]);
        }

        if(l<mlp->L-1)
        {
            for (j=0;j<mlp->J[l+1];j++)
                for(i=0;i<mlp->J[l];i++)
                    mlp->x[l+1][j][i]=mlp->Out[l][i];
        }
    }

    //-------формирование выходного вектора--------------------

    for (l=(mlp->L-1),j=0;j<mlp->Y;j++)
        mlp->Yout[j]=mlp->Out[l][j];

}

//----------------------------------------------------------------------------------------------------

void training_MLP(MLP *mlp)
{
    struct trVec *tV;
    FILE *FtrDATA;
    char marker[4]="___"; //маркер  файла с данными
    int M;                // размерности обучающей выборки
    int X;                // размерность входного обучающего вектора
    int Y;                // размерность выходного обучающего вектора

    double E;             // среднеквадратическая ошибка
    double epsilon=0.01;   // пороговое значение ошибки
    double Mu=0.1;       // коэфицент  скорости обучения
    int N_era=100000;       // колличество эпох обучения

    double **sigma;        // расчетный коэф. ошибки нейрона
    double dw;             // величина корректировки веса нейрона



    // ---вспомогательные переменные

    char *temp;
    char *file_name;
    int l;
    int j;
    int i;
    int k;
    int m;
    double tempSigma;

    FtrDATA=NULL;
    temp=calloc(15,sizeof(char));
    file_name=calloc(15,sizeof(char));

    // ------------------------открытие файла c обучающими данными----------------------
    while (!((FtrDATA!=NULL)&&(strcmp(marker,"trD\0")==0)))  //--вводим имя файла---
     {
     printf("Enter file name:");
     fgets(temp,15,stdin);

 /*    if (strcmp(temp3,"ex\n")==0) //выход из процедуры
         return;
  */
     strncpy(file_name,temp,strlen(temp)-1);
     FtrDATA=fopen(file_name,"rb");
     if (FtrDATA==NULL)
     {
         printf("Error! Can not open file with train data!!!\n");
         temp=realloc(NULL,15*sizeof(char));
     }
     else
     {
         fread(&marker,sizeof(marker),1,FtrDATA);
         if (strcmp(marker,"trD\0")!=0)
         {
             printf("it`s a not file with training data\n");
             fclose(FtrDATA);
         }
     }
      //rewind(stdin); //для Linux
     fflush(stdin); //для Windows
     }

    fread(&X,sizeof(int),1,FtrDATA);
    fread(&Y,sizeof(int),1,FtrDATA);
    fread(&M,sizeof(int),1,FtrDATA);

    if((X!=mlp->X)&&(Y!=mlp->Y))
    {
        printf("training data file not for your initialize configuration of MLP\n");
        fclose(FtrDATA);
        free(file_name);
        return;
    }

    tV=calloc(M,sizeof(struct trVec));

    for (i=0;i<M;i++)//выделяем место в куче
    {
        tV[i].Xin=calloc(X,sizeof(double));
        tV[i].Yout=calloc(Y,sizeof(double));
    }

    for (i=0;i<M;i++)// цикл чтения  данных обучающего вектора
    {
        for (j=0;j<X;j++)
            fread(&(tV[i].Xin[j]),sizeof(double),1,FtrDATA);
        for (j=0;j<Y;j++)
            fread(&(tV[i].Yout[j]),sizeof(double),1,FtrDATA);
    }



#ifdef DEBUG
    for (i=0;i<M;i++)
    {
        for (j=0;j<X;j++)
            printf ("x  %f   ",tV[i].Xin[j]);
        printf("\n");
        for (j=0;j<Y;j++)
            printf ("           y  %f   ",tV[i].Yout[j]);
        printf("\n");
    }
       printf("++++++++++++++++++++++++\n");
#endif

    //----------нормалицация входного вектора------------

    if (mlp->FofA==1)
        rating01(tV,M,X,Y,mlp);
    else
        rating11(tV,M,X,Y,mlp);


#ifdef DEBUG
   for (i=0;i<M;i++)
   {
       for (j=0;j<X;j++)
           printf ("x  %f   ",tV[i].Xin[j]);
       printf("\n");
       for (j=0;j<Y;j++)
           printf ("            y  %f   ",tV[i].Yout[j]);
       printf("\n");
   }
   printf("++++++++++++++++++++++++\n");
#endif

    //----инициализация массива коэф. ошибки
    sigma=malloc(mlp->L*sizeof(double*));
    for (l=1;l<mlp->L;l++)
        sigma[l]=calloc((mlp->J[l]),sizeof(double));

     mixing_trVec(tV,M,X,Y);

#ifdef DEBUG
   for (i=0;i<M;i++)
   {
       for (j=0;j<X;j++)
           printf ("x  %f   ",tV[i].Xin[j]);
       printf("\n");
       for (j=0;j<Y;j++)
           printf ("            y  %f   ",tV[i].Yout[j]);
       printf("\n");
   }
   printf("++++++++++++++++++++++++\n");
#endif

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    while (((E=RMS_error(mlp,tV,M,X,Y))>epsilon)&&(N_era!=0))
    {

        //миксуем обучающий вектор в случайном порядке
        mixing_trVec(tV,M,X,Y);

        for (k=0;k<M;k++)
        {
            basic_calc_MLP(mlp,tV[k].Xin); //пря мой проход сети

            //.....расчет коэфициентов ошибки нейронов выходного слоя.....
            for (l=mlp->L-1,j=0;j<mlp->J[l];j++)
            {
                sigma[l][j]=(tV[k].Yout[j]-mlp->Out[l][j])*(*Der_funk_of_act)(mlp->NET[l][j]-mlp->O[l][j]);
            }

            //... расчет коэфициентов ошибки нейронов скрытых слоев....

            for (l=mlp->L-2;l>0;l--)
            {
                for (j=0;j<mlp->J[l];j++)
                {
                    tempSigma=0;
                    for (m=0;m<mlp->J[l+1];m++)
                        tempSigma+=sigma[l+1][m] * mlp->w[l+1][m][j];

                    sigma[l][j]=tempSigma*(*Der_funk_of_act)(mlp->NET[l][j]-mlp->O[l][j]);
                }
           }
            //.....корректировка весов выходного слоя.....
            for (l=mlp->L-1,j=0;j<mlp->J[l];j++)
            {
                dw=0;
                for (i=0;i<mlp->J[l-1];i++)
                {
                    dw=Mu*sigma[l][j]*mlp->Out[l-1][i];
                    mlp->w[l][j][i]=mlp->w[l][j][i]+dw;
                }
            }
            //... корректировка весов скрытых слоев....
            for (l=mlp->L-2;l>0;l--)
            {
                for (j=0;j<mlp->J[l];j++)
                {
                    dw=0;
                    for (i=0;i<mlp->J[l-1];i++)
                    {
                        dw=Mu*sigma[l][j]*mlp->Out[l-1][i];
                        mlp->w[l][j][i]=mlp->w[l][j][i]+dw;
                    }
                }
            }

        }
        N_era-=1;
        printf("E=%f\n",E);
      }

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    //освобождаем место в дин.памяти
    fclose(FtrDATA);
    free(file_name);
    for (i=0;i<M;i++)
    {
        free(tV[i].Xin);
        free(tV[i].Yout);
    }
    free(tV);

    for (l=1;l<mlp->L;l++)
        free (sigma[l]);
    free (sigma);

}


//----------------------------------------------------------------------------------------------------
void creat_train_data_file (MLP* mlp)
{
    struct trVec *tV;
    FILE *FtrDATA;
    char marker[4]="trD\0";//маркер  файла с данными
    int M;// размерности обучающей выборки
    int X;
    int Y;

    // ---вспомогательные переменные
    char *temp;
    char *file_name;
    int i;
    int j;
    int flag;
    double temp1;


  //--------------------------------------------------

    X=mlp->X;
    Y=mlp->Y;
    FtrDATA=NULL;
    temp=calloc(10,sizeof(char));
    file_name=calloc(10,sizeof(char));

    //--вводим размерность обучающей выборки
    flag=0;
    printf("\n");
    printf("enter capacity of train data=");
    while(!flag)
    {
        fgets(temp,3,stdin);
        flag=num_control(temp,1000,1,0,0);
        if (flag==0)  printf("Enter data not correct! Try again! \n");
        else M=atof(temp);
        //rewind(stdin); //для Linux
        fflush(stdin); //для Windows
    }
    printf("\n");

    // выделяем в куче мето под обучающий выектор
    tV =(struct trVec *) calloc(M,sizeof(struct trVec));
    for (i=0;i<M;i++)
    {
        tV[i].Xin=calloc(X,sizeof(double));
        tV[i].Yout=calloc(Y,sizeof(double));
    }

    //заполняем весь обучающий вектор
    for (i=0;i<M;i++)
    {
        printf("   %d train vector\n",i+1);
        for (j=0;j<X;j++)
        {
            flag=0;
            printf("\n");
            printf("X%d=",j+1);
            while((!flag)||(flag==-1))
            {
                fgets(temp,5,stdin);
                flag=num_control(temp,1000,-1000,1,1);
                if ((flag==0)||(flag==-1))  printf("Enter data not correct! Try again! \n");
                else tV[i].Xin[j]=atof(temp);
                //rewind(stdin); //для Linux
                fflush(stdin); //для Windows
            }
            printf("\n");
        }


        for (j=0;j<Y;j++)
        {
            flag=0;
            printf("\n");
            printf("Y%d=",j+1);
            while((!flag)||(flag==-1))
            {
                fgets(temp,5,stdin);
                flag=num_control(temp,1000,-1000,1,1);
                if ((flag==0)||(flag==-1))  printf("Enter data not correct! Try again! \n");
                else tV[i].Yout[j]=atof(temp);
                //rewind(stdin); //для Linux
                fflush(stdin); //для Windows
            }
            printf("\n");
        }
    }


   // ------------------------запись в файл----------------------
    while (FtrDATA==NULL)  //--вводим имя файла---
    {
    printf("Enter file name:");
    fgets(temp,15,stdin);

/*    if (strcmp(temp3,"ex\n")==0) //выход из процедуры
        return;
 */
    strncpy(file_name,temp,strlen(temp)-1);
    FtrDATA=fopen(file_name,"wb");
    if (FtrDATA==NULL)
        printf("Error! Can not open file with train data!!!\n");
    //rewind(stdin); //для Linux
    fflush(stdin); //для Windows
    }


    fwrite(marker,sizeof(marker),1,FtrDATA);// помечаем  обучающего файла
    fwrite(&X,sizeof(int),1,FtrDATA);//запись размерности вх вектора
    fwrite(&Y,sizeof(int),1,FtrDATA);//запись размерности выходного выктора
    fwrite(&M,sizeof(int),1,FtrDATA); //запись размерности обучающей выборки

    for (i=0;i<M;i++)// цикл записи данных обучающего вектора
    {

        for (j=0;j<X;j++)
        {
           fwrite(&(tV[i].Xin[j]),sizeof(double),1,FtrDATA);
           temp1=tV[i].Xin[j];
        }


        for (j=0;j<Y;j++)
            fwrite(&(tV[i].Yout[j]),sizeof(double),1,FtrDATA);

    }

    if (FtrDATA!=NULL)
    {
        printf("\n");
        printf("train data file- %s was create successfull",file_name);
        printf("\n");
    }


    //---освобождаем память----
    fclose(FtrDATA);
    free(temp);
    for (i=0;i<M;i++)
    {
        for (j=0;j<X;j++)
        {
           free(tV[i].Xin);
           free(tV[i].Yout);
        }


    }
    free(file_name);
}

//----------------------------------------------------------------------------------------------------
void output_train_data(MLP* mlp)
{
    struct trVec *tV;
    FILE *FtrDATA;
    char marker[4]="___";//маркер  файла с данными
    int M;// размерности обучающей выборки
    int X;
    int Y;

    // ---вспомогательные переменные

    char *temp;
    char *file_name;
    int i;
    int j;


    FtrDATA=NULL;
    temp=calloc(15,sizeof(char));
    file_name=calloc(15,sizeof(char));

    // ------------------------открытие файла ----------------------
    while (!((FtrDATA!=NULL)&&(strcmp(marker,"trD\0")==0)))  //--вводим имя файла---
     {
     printf("Enter file name:");
     fgets(temp,15,stdin);

 /*    if (strcmp(temp3,"ex\n")==0) //выход из процедуры
         return;
  */
     strncpy(file_name,temp,strlen(temp)-1);
     FtrDATA=fopen(file_name,"rb");
     if (FtrDATA==NULL)
     {
         printf("Error! Can not open file with train data!!!\n");
         temp=realloc(NULL,15*sizeof(char));
     }
     else
     {
         fread(&marker,sizeof(marker),1,FtrDATA);
         if (strcmp(marker,"trD\0")!=0)
         {
             printf("it`s a not file with training data\n");
             fclose(FtrDATA);
         }
     }
      //rewind(stdin); //для Linux
     fflush(stdin); //для Windows
     }

    fread(&X,sizeof(int),1,FtrDATA);
    fread(&Y,sizeof(int),1,FtrDATA);
    fread(&M,sizeof(int),1,FtrDATA);

    if((X!=mlp->X)&&(Y!=mlp->Y))
    {
        printf("training data file not for your initialize configuration of MLP\n");
        fclose(FtrDATA);
        free(file_name);
        return;
    }

    tV=calloc(M,sizeof(struct trVec));

    for (i=0;i<M;i++)//выделяем место в куче
    {
        tV[i].Xin=calloc(X,sizeof(double));
        tV[i].Yout=calloc(Y,sizeof(double));
    }

    for (i=0;i<M;i++)// цикл чтения  данных обучающего вектора
    {

        for (j=0;j<X;j++)
        {
            fread(&(tV[i].Xin[j]),sizeof(double),1,FtrDATA);
        }


        for (j=0;j<Y;j++)
        {
            fread(&(tV[i].Yout[j]),sizeof(double),1,FtrDATA);
        }
    }


    for (i=0;i<M;i++)// выводим значения данных обучающего вектора
    {

        printf("     %d training vector\n",i+1);
        for (j=0;j<X;j++)
            printf("Xin%d=%f   ",j+1,tV[i].Xin[j]);

        for (j=0;j<Y;j++)
            printf("Yout%d=%f   ",j+1,tV[i].Yout[j]);
        printf("\n");

    }




    fclose(FtrDATA);
    free(file_name);
    for (i=0;i<M;i++)//освобождаем место в куче
    {
        free(tV[i].Xin);
        free(tV[i].Yout);
    }
    free(tV);

}
//----------------------------------------------------------------------------------------------------

double RMS_error (MLP* mlp,struct trVec * tV, int M,int X, int Y)
{
    double E;
    double *tempX;
    double *tempY;
    int i;
    int j;

    tempX=calloc(X,sizeof(double));
    tempY=calloc(Y,sizeof(double));

    //----------расчет среднеквадратической ошибки-------------
    E=0;
    for (i=0;i<M;i++)
    {
        for (j=0;j<X;j++)
            tempX[j]=tV[i].Xin[j];
        basic_calc_MLP(mlp,tempX);// расчет выхода по обуч. вх вектору как тестовому
                                  // прямой проход сети
        for (j=0;j<Y;j++)
            tempY[j]=mlp->Yout[j];

        // расчитаем ошибку на шаге i
        for (j=0;j<Y;j++)
            E+=pow((tV[i].Yout[j]-tempY[j]),2);
    }

    free (tempX);
    free (tempY);

    return 0.5*E;

    // требуеться упростить и оптимизировать, убрав tempX и tempY


}
//----------------------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------------------
// ф-ия проверки на число(целое/вещ.определяется перем DorI=0/1)
// принадлежность интервалу (min;max)
// положительное/отрицательным перем SorUS=0/1 (min задавать <0)


int num_control(char *temp, int max,int min, int DorI, int SorUS )
 {
    int flag,counter,i;
    flag=0;
    counter=0;


    if (strcmp(temp,"ex\n")==0) // условие выхода из процедуры
        return -1;

    if(!SorUS)
    {
        for (i=0;i<strlen(temp)-1;i++)
        {
            if (!DorI)
            {
                if   (!(isdigit(temp[i])))
                    flag=1;
            }
            else
            {
                if ((temp[i]=='.'))
                    counter++;
                if   (!((isdigit(temp[i])||(temp[i]=='.'))&&(counter<2)))
                      flag=1;
            }

        }
    }
    else
    {
        if(temp[0]=='-')
        {
            for (i=1;i<strlen(temp)-1;i++)
            {
                if (!DorI)
                {
                    if   (!(isdigit(temp[i])))
                        flag=1;
                }
                else
                {
                    if ((temp[i]=='.'))
                        counter++;
                    if   (!((isdigit(temp[i])||(temp[i]=='.'))&&(counter<2)))
                          flag=1;
                }

            }
        }
        else
        {
            for (i=0;i<strlen(temp)-1;i++)
            {
                if (!DorI)
                {
                    if   (!(isdigit(temp[i])))
                        flag=1;
                }
                else
                {
                    if ((temp[i]=='.'))
                        counter++;
                    if   (!((isdigit(temp[i])||(temp[i]=='.'))&&(counter<2)))
                          flag=1;
                }

            }


        }


     }

    if((flag)||(atoi(temp)>(max))||(atoi(temp)<(min)))
        return 0;
    else
        return 1;

 }



//----------------------------------------------------------------------------------------------------
double sigmoid (double net)
{
    double out;
    out = 1/(1+exp(-net));
    return out;
}


//----------------------------------------------------------------------------------------------------
double gip_tan (double net)
{
    double out;
    out = (exp(net)-exp(-net))/(exp(net)+exp(-net));
    return out;
}

//----------------------------------------------------------------------------------------------------
double Der_sigmoid (double X)
{
    double out;
    out=sigmoid(X)*(1-sigmoid(X));
    return out;

}

//----------------------------------------------------------------------------------------------------
double Der_gip_tan (double X)
{
    double out;
    out =(1-(gip_tan(X)*gip_tan(X)));
    return out;
}

//----------------------------------------------------------------------------------------------------
void rating01 (struct trVec *tV, int M, int X, int Y, MLP *mlp)
{
    double minX;
    double maxX;
    double deltaX;
    double minY;
    double maxY;
    double deltaY;
    int i;
    int j;

    maxX=tV[0].Xin[0];
    minX=tV[0].Xin[0];
    maxY=tV[0].Yout[0];
    minY=tV[0].Yout[0];

    for (i=0;i<M;i++)
    {
        for (j=0;j<X;j++)
        {
            if (maxX<tV[i].Xin[j])
                maxX=tV[i].Xin[j];
            if (minX>tV[i].Xin[j])
                minX=tV[i].Xin[j];
        }

        for (j=0;j<Y;j++)
        {
            if (maxY<tV[i].Yout[j])
                maxY=tV[i].Yout[j];
            if (minY>tV[i].Yout[j])
                minY=tV[i].Yout[j];
        }
     }

    deltaX=maxX-minX;
    deltaY=maxY-minY;


    for (i=0;i<M;i++)
    {
        for (j=0;j<X;j++)
            tV[i].Xin[j]=(tV[i].Xin[j]-minX)/deltaX;
        for (j=0;j<Y;j++)
            tV[i].Yout[j]=(tV[i].Yout[j]-minY)/deltaY;
    }

    mlp->alphaX=minX;
    mlp->alphaY=minY;
    mlp->deltaY=deltaY;
    mlp->deltaX=deltaX;

}

//----------------------------------------------------------------------------------------------------
void rating11 (struct trVec *tV, int M, int X, int Y,MLP *mlp)
{
    double minX;
    double maxX;
    double deltaX;
    double minY;
    double maxY;
    double deltaY;
    int i;
    int j;

    maxX=tV[0].Xin[0];
    minX=tV[0].Xin[0];
    maxY=tV[0].Yout[0];
    minY=tV[0].Yout[0];

    for (i=0;i<M;i++)
    {
        for (j=0;j<X;j++)
        {
            if (maxX<tV[i].Xin[j])
                maxX=tV[i].Xin[j];
            if (minX>tV[i].Xin[j])
                minX=tV[i].Xin[j];
        }
        for (j=0;j<Y;j++)
        {
            if (maxY<tV[i].Yout[j])
                maxY=tV[i].Yout[j];
            if (minY>tV[i].Yout[j])
                minY=tV[i].Yout[j];
        }

     }

    deltaX=maxX-minX;
    deltaY=maxY-minY;

    for (i=0;i<M;i++)
    {
        for (j=0;j<X;j++)
            tV[i].Xin[j]=2*(tV[i].Xin[j]-minX)/deltaX-1;
        for (j=0;j<Y;j++)
            tV[i].Yout[j]=2*(tV[i].Yout[j]-minY)/deltaY-1;

    }
    mlp->alphaX=minX;
    mlp->alphaY=minY;
    mlp->deltaX=deltaX;
    mlp->deltaY=deltaY;
}
//----------------------------------------------------------------------------------------------------
void mixing_trVec(struct trVec * tV,int M, int X, int Y)
{
    int i;
    int j;
    int k;
    int m;
    double temp;

    temp=0;
    srand(time(NULL));

    for (i=0;i<M;i++)
    {
        j=rand()%M;
        k=rand()%M;
        for (m=0;m<X;m++)
         {
            temp=tV[j].Xin[m];
            tV[j].Xin[m]=tV[k].Xin[m];
            tV[k].Xin[m]=temp;
         }
        for (m=0;m<Y;m++)
         {
            temp=tV[j].Yout[m];
            tV[j].Yout[m]=tV[k].Yout[m];
            tV[k].Yout[m]=temp;
         }
    }
}



