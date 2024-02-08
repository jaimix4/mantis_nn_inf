/*
 * run_orig.c
 *
 *  Created on: Aug 16, 2017
 *      Author: w.zholobenko
 *
 *  For given Te / ne from the command line,
 *  returns RTe / Rne (line intensity ratios)
 *  in form. II (neglecting recombination). B set.
 */

 /*

  * Created on: 14 Oct 2022
  * Author: Jaime Caballero

  * Takes combinations of Te and ne from a csv files
  * and prints a csv file with the values of the line intensity ratios
  * 728/706     728/668


 */

 /*

  * Created on: 5 Feb 2024
  * Author: Jaime Caballero

  * Takes Te, ne, net from a csv files
  * and prints a csv file with the values of emission for
  * 728, 706, 668 He; ignoring recombination emission 


 */

#include <stdio.h>
#include <math.h>
#include "hecrm.h"
#include <stdlib.h>
#include <string.h>

#define NUMNE 1 /* number of electron density values in a single `hecrmodel' call */
/*int main(int argc, char *argv[]);*/

#define MAXCHAR 60

int main(int argc, char *argv[]) /* int argc, char *argv[] */
{
	crmodel_prm prmdata;
	rate_coefficient rate;

	cr_rate_coefficient crrate[NUMNE];
	population_coefficient popcoe[NUMNE];
	cooling_rate_coefficient coolrate[NUMNE];


	/* initialize variables */

	make_rate_arrays(&rate);
	init_rate_arrays(&rate);

	/* magnetic field strength [T] */

	prmdata.mfield = 1.4;

	/* electron temperature [eV] */

	prmdata.te = 20.0 ;/*  atof( argv[1] );  */

	/* electron density [cm^{-3}] */

	prmdata.ne = gsl_vector_calloc(NUMNE);
	VS(prmdata.ne, 0, 2e13/2); /*  1e19 m^{-3}   *//*  atof( argv[2] )  */


  FILE *Tene;

  FILE *out;

  double Te;

  double ne;

  // double nHe;

  char buffer[MAXCHAR];

  //Tene = fopen("check_Heratios_76_2.csv","r");

  //out = fopen("OUT_check_Heratios_76_2.csv", "w");


  // printf(argv[1]);
  // printf(argv[2]);

  Tene = fopen(argv[1],"r");

  out = fopen(argv[2], "w");

  // 

  int u = 0;


  while (fgets(buffer, MAXCHAR, Tene)) {


    // If you need all the values in a row
    char *token = strtok(buffer, ",");

    while (token) {
        // Just printing each integer here but handle as needed


        if (u % 2 == 0) {

        Te = atof(token);


        //printf("%10.8e\n", Te);

        } else {

        ne = atof(token);


        //printf("%10.8e\n", ne);

        
				if ((Te != 0.0 && ne != 0.0) && Te > 0.04 && Te < 250 && ne > 1.0e15 && ne < 2.5e20) {

				prmdata.te = Te ;

				VS(prmdata.ne, 0, ne/1e6);

        hecrmodel(&prmdata, &rate, popcoe, crrate, coolrate);

        //double RTe = MG(rate.a, 5, 3) / MG(rate.a, 6, 4) * popcoe[0].rr1[5] / popcoe[0].rr1[6];
        //double Rne = MG(rate.a, 5, 3) / MG(rate.a, 9, 3) * popcoe[0].rr1[5] / popcoe[0].rr1[9];

        double pec_exc_He728 = MG(rate.a, 5, 3) * popcoe[0].rr1[5];
        double pec_exc_He706 = MG(rate.a, 6, 4) * popcoe[0].rr1[6];
        double pec_exc_He668 = MG(rate.a, 9, 3) * popcoe[0].rr1[9];

        double pec_rec_He728 = MG(rate.a, 5, 3) * popcoe[0].rr0[5];
        double pec_rec_He706 = MG(rate.a, 6, 4) * popcoe[0].rr0[6];
        double pec_rec_He668 = MG(rate.a, 9, 3) * popcoe[0].rr0[9];


        // here you have to compute excitation PECs and recombination PECs of 728, 706, 668
        // lets see how I do that 



        //fprintf(stdout, "R_Te 728 / 706   = %10.8e\n", RTe);
        //fprintf(stdout, "R_ne 728 / 668   = %10.8e\n", Rne);

        fprintf(out, "%10.15e, %10.15e, %10.15e, %10.15e, %10.15e, %10.15e, %10.15e, %10.15e\n", Te, ne, pec_exc_He728, pec_exc_He706, pec_exc_He668, pec_rec_He728, pec_rec_He706, pec_rec_He668);

			}  else {

				fprintf(out, "NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN\n");

			}

        }

            token = strtok(NULL, ",");
            u++;

            if (u % 100 == 0) {

                printf("%d\n", u/2);

            }

            }
    }

  printf("%d", u/2);


fclose(Tene);
fclose(out);

	printf("\n DONE");

	return 0;



}
