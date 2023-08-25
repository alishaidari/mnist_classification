#include <stdio.h>
#include <stdlib.h> 
#include <omp.h>
#include <limits.h>
#include "matrix.h"
#include "maxheap.h"

/* find the k indices of the training images closest to the test image */
/* in sorted order from closest to farthest. */
/* Use the provided maxheap code to make this function efficient. */
void find_knearest (matrix_type* train_images_ptr, matrix_row_type* test_image_ptr,
		    int k, int* knearest) {
    /* replace the following "stub" code with code that finds the indices of the */
    /* k nearest neighbors in sorted order from closest to farthest. */
    
    int i, j, num_images;
    num_images = train_images_ptr->num_rows;
    int distance_arr[num_images];
    matrix_row_type current_train_image;
    for (j=0;j<train_images_ptr->num_rows;j++){
        matrix_get_row(train_images_ptr, &current_train_image, j);
        distance_arr[j] = matrix_row_dist_sq(&current_train_image, test_image_ptr);
    }
  
    for (i=0;i<k;i++) {
        int distance_min = INT_MAX;
        int index_of_min = 0;
        for(j=0;j<num_images;j++){
            if(distance_arr[j] < distance_min){
                distance_min = distance_arr[j];
                index_of_min = j;
            }
        }
        distance_arr[index_of_min] = INT_MAX - 1; 
        knearest[i] = index_of_min;
    }
}
int main (int argc, char** argv) {
    
    /* get k, start_index, num_to_test, num_threads, and (time_only) from command line */
    if ((argc != 5) && (argc != 6)) {
	printf ("Command usage : %s %s %s %s %s %s\n",argv[0],"k","start_index",
		"num_to_test","num_threads","(time_only)");
	return 1;
    }
    int k = atoi(argv[1]);
    int start_index = atoi(argv[2]);
    int num_to_test = atoi(argv[3]);
    if (start_index+num_to_test > 10000) {
	printf ("error : start_index+num_to_test > 10000\n");
	exit(1);
    }
    int num_threads = atoi(argv[4]);
    omp_set_num_threads(num_threads);
    int time_only = 0;
    if (argc == 6) {
	time_only = 1;
    }
    
    /* read in the mnist training set of 60000 images and labels */
    int num_train = 60000;
    matrix_type train_images, train_labels;
    matrix_init (&train_images,num_train,784);
    matrix_read_bin(&train_images,"train-images-idx3-ubyte",16);
    matrix_init (&train_labels,num_train,1);
    matrix_read_bin(&train_labels,"train-labels-idx1-ubyte",8);
    
    /* read in the mnist test set of 10000 images */
    int num_test = 10000;
    matrix_type test_images, test_labels;
    matrix_init (&test_images,num_test,784);
    matrix_read_bin(&test_images,"t10k-images-idx3-ubyte",16);
    matrix_init (&test_labels,num_test,1);
    matrix_read_bin(&test_labels,"t10k-labels-idx1-ubyte",8);
    
    /* start the timer */
    double start, end;
    start = omp_get_wtime();
    
    /* for each test image, find the k nearest training images */
    int i,j;
    matrix_row_type test_image;
    int knearest[k];
    for (i = start_index;i<start_index+num_to_test;i++) {
	matrix_get_row(&test_images,&test_image,i);
	find_knearest (&train_images,&test_image,k,knearest);
	if (time_only == 0) {
	    printf ("test index : %d, test label : %d\n",i,test_labels.data_ptr[i]);
	    printf (" training indices :");
	    for (j = 0;j<k;j++) {
		printf (" %d",knearest[j]);
	    }
	    printf (", training labels :");
	    for (j = 0;j<k;j++) {
		printf (" %d",train_labels.data_ptr[knearest[j]]);
	    }
	    printf ("\n");
	}
    }
    
    /* stop timer */
    end = omp_get_wtime();
    
    /* print the wall time used */
    printf ("wall time used = %g sec\n",end-start);
    //printf("the training image set has %d columns and %d rows. \n", train_images.num_cols, train_images.num_rows); 
    /* free up the training and test data sets */
    matrix_deinit(&train_images);
    matrix_deinit(&test_images);
    matrix_deinit(&train_labels);
    matrix_deinit(&test_labels);
    
    return 0;
}

