/* **************************************************************** */
/* *************  Calcular la matriz de coocurrencia  ************* */
/* **************************************************************** */

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <string>
#include <cmath>

using namespace cv;
using namespace std;

double suma2 = 0.0;
double contraste = 0.0;
double uniformidad = 0.0;
double entropia = 0.0;
double homogeneidad = 0.0;
double sumaNorm = 0.0;
double mr = 0.0;
double mc = 0.0;
double sigma_r = 0.0;
double correlacion = 0.0;

int main() {

    Mat imagen, imagenGris, dividida, coocurrencia, coocSimetrica, coNormalizada;
    namedWindow("window", 0);
    namedWindow("window1", 0);
    imagen = imread("zebra.png");
    cvtColor(imagen, imagenGris, CV_RGB2GRAY);
    dividida = imagenGris/35;
    coocurrencia = Mat::zeros(8, 8, CV_8UC1);
    coocSimetrica = Mat::zeros(8, 8, CV_8UC1);
    coNormalizada = Mat::zeros(8, 8, CV_64FC1);

    /* Ciclo for para hallar la matriz de co-ocurrencia */
    for (int j = 0; j<imagenGris.rows; j++) {
        for (int i = 0; i<imagenGris.cols - 1; i++) {
            coocurrencia.at<uchar>(dividida.at<uchar>(j, i), dividida.at<uchar>(j, i + 1)) += 1;
        }
    }

    //Imprimir la matriz de co-ocurrencia
    for (int j = 0; j<coocurrencia.rows; j++) {
        for (int i = 0; i<coocurrencia.cols - 1; i++) {
            printf("%d\t", coocurrencia.at<uchar>(j, i));
        }
        printf("\n");
    }
    printf("\n");

    //Hallar la matriz simetrica de co-ocurrencia
    coocSimetrica = coocurrencia + coocurrencia.t();
    suma2 = sum(coocSimetrica)[0];
    printf("%f\n\n Normalizada:\n", suma2);

    //Normalizar la matriz de co-ocurrencia
    for (int j = 0; j < coocSimetrica.rows; j++) {
        for (int i = 0; i < coocSimetrica.cols; i++) {
            coNormalizada.at<double>(j, i) = double(coocSimetrica.at<uchar>(j, i) / suma2);
            printf("%f ", coNormalizada.at<double>(j, i));
            //suma_norm += coNormalizada.at<double>(j, i); 
        }
        printf("\n");
    }
    printf("\n");

    //Hallar mr
    for (double j = 0; j < coNormalizada.rows; j++) {
        for (double i = 0; i < coNormalizada.cols; i++) {
            mr += (j+1)*coNormalizada.at<double>(j, i);
            mc += (j+1)*coNormalizada.at<double>(i, j);
        }
    }

    //Hallar sigma_r == sigma_c
    for (double j = 0; j < coNormalizada.rows; j++) {
        for (double i = 0; i < coNormalizada.cols; i++) {
            sigma_r += pow(j+1-mr,2)*coNormalizada.at<double>(j, i);
        }
    }
    sigma_r = sqrt(sigma_r);

    printf("\nContraste: ");

    //Hallar descriptores de 2do orden
    for (double j = 0; j < coNormalizada.rows; j++) {
        for (double i = 0; i < coNormalizada.cols; i++) {
            contraste += coNormalizada.at<double>(j, i)*pow(i-j, 2);
            uniformidad += pow(coNormalizada.at<double>(j, i), 2);
            if(coNormalizada.at<double>(j, i) != 0.0){
                entropia += (-1)*coNormalizada.at<double>(j, i)*log2(coNormalizada.at<double>(j, i));
            }
            homogeneidad += coNormalizada.at<double>(j, i) / (1 + abs(j - i));
            correlacion += (i+1-mr)*(j+1-mr)*coNormalizada.at<double>(j, i)/pow(sigma_r, 2);
        }
    }

    printf("%f\n", contraste);
    printf("Uniformidad: %f\n", uniformidad);
    printf("Entropia: %f\n", entropia);
    printf("Homogeneidad: %f\n", homogeneidad);
    printf("correlacion: %f\n", correlacion);
    imshow("window", imagen);
    imshow("window1", coNormalizada);
    waitKey(0);
    getchar();
    return 0;
}

