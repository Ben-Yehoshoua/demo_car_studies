/// @file module_driving.cpp
/// @brief Implémentation des fonctions de conduite pour la voiture.
/// 
/// Ce fichier configure les broches de contrôle des moteurs pour effectuer
/// les mouvements de la voiture : avancer, reculer, tourner, s'arrêter.

#include "Arduino.h"
#include "module_driving.hpp"

/// @brief Broche d'activation du moteur A
const int Moteurs_Active_1 = 9;

/// @brief Broche arrière du moteur A
const int Moteurs_En_Avant_1 = 7;

/// @brief Broche avant du moteur A
const int Moteurs_En_Arriere_1 = 10;

/// @brief Broche d'activation du moteur B
const int Moteurs_Active_2 = 11;

/// @brief Broche arrière du moteur B
const int Moteurs_En_Avant_2 = 12;

/// @brief Broche avant du moteur B
const int Moteurs_En_Arriere_2 = 13;

/// @brief Durée de rotation des moteurs (en ms)
#define ROTATION_MOTEURS_TEMPS (1000U)

/**
 * @brief Fait avancer la voiture.
 * 
 * Active les moteurs pour faire avancer la voiture pendant un temps défini.
 * 
 * @return Code de retour 4.
 */
int fonction_conduite_voiture_avance(void)
{
	// Configuration des broches
	pinMode(Moteurs_Active_1, OUTPUT);
	pinMode(Moteurs_En_Arriere_1, OUTPUT);
	pinMode(Moteurs_En_Arriere_2, OUTPUT);

	pinMode(Moteurs_Active_1, OUTPUT);
	pinMode(Moteurs_En_Avant_1, OUTPUT);
	pinMode(Moteurs_En_Avant_2, OUTPUT);
	
	// Commande des moteurs pour avancer
	digitalWrite(Moteurs_En_Arriere_1, LOW);
	digitalWrite(Moteurs_En_Arriere_2, LOW);
	digitalWrite(Moteurs_En_Avant_1, HIGH);
	digitalWrite(Moteurs_En_Avant_2, HIGH);
	analogWrite(Moteurs_Active_1, 255);  
	analogWrite(Moteurs_Active_2, 255);  
	
	delay(ROTATION_MOTEURS_TEMPS); 
	
	// Arrêter les moteurs
	analogWrite(Moteurs_Active_1, 0);
	analogWrite(Moteurs_Active_2, 0);
	
	return 4;
}

/**
 * @brief Fait reculer la voiture.
 * 
 * Active les moteurs en sens inverse pour reculer pendant un temps défini.
 * 
 * @return Code de retour 5.
 */
int fonction_conduite_voiture_recule(void)
{
	// Configuration des broches
	pinMode(Moteurs_Active_1, OUTPUT);
	pinMode(Moteurs_En_Arriere_1, OUTPUT);
	pinMode(Moteurs_En_Arriere_2, OUTPUT);

	pinMode(Moteurs_Active_1, OUTPUT);
	pinMode(Moteurs_En_Avant_1, OUTPUT);
	pinMode(Moteurs_En_Avant_2, OUTPUT);
	
	// Commande des moteurs pour reculer
	digitalWrite(Moteurs_En_Arriere_1, HIGH);
	digitalWrite(Moteurs_En_Arriere_2, HIGH);
	digitalWrite(Moteurs_En_Avant_1, LOW);
	digitalWrite(Moteurs_En_Avant_2, LOW);
	analogWrite(Moteurs_Active_1, 255);  
	analogWrite(Moteurs_Active_2, 255);  
	
	delay(ROTATION_MOTEURS_TEMPS); 
	
	// Arrêter les moteurs
	analogWrite(Moteurs_Active_1, 0);
	analogWrite(Moteurs_Active_2, 0);
	
	return 5;
}

/**
 * @brief Fait tourner la voiture à droite.
 * 
 * Configure les moteurs pour un virage à droite pendant un temps défini.
 * 
 * @return Code de retour 6.
 */
int fonction_conduite_voiture_tourne_a_droite(void)
{
	// Configuration des broches
	pinMode(Moteurs_Active_1, OUTPUT);
	pinMode(Moteurs_En_Arriere_1, OUTPUT);
	pinMode(Moteurs_En_Arriere_2, OUTPUT);

	pinMode(Moteurs_Active_1, OUTPUT);
	pinMode(Moteurs_En_Avant_1, OUTPUT);
	pinMode(Moteurs_En_Avant_2, OUTPUT);
	
	// Commande des moteurs pour tourner à droite
	digitalWrite(Moteurs_En_Arriere_1, LOW);
	digitalWrite(Moteurs_En_Arriere_2, HIGH);
	digitalWrite(Moteurs_En_Avant_1, HIGH);
	digitalWrite(Moteurs_En_Avant_2, LOW);
	analogWrite(Moteurs_Active_1, 255);  
	analogWrite(Moteurs_Active_2, 255);  
	
	delay(ROTATION_MOTEURS_TEMPS); 
	
	// Arrêter les moteurs
	analogWrite(Moteurs_Active_1, 0);
	analogWrite(Moteurs_Active_2, 0);
	
	return 6;
}

/**
 * @brief Fait tourner la voiture à gauche.
 * 
 * Configure les moteurs pour un virage à gauche pendant un temps défini.
 * 
 * @return Code de retour 7.
 */
int fonction_conduite_voiture_tourne_a_gauche(void)
{
	// Configuration des broches
	pinMode(Moteurs_Active_1, OUTPUT);
	pinMode(Moteurs_En_Arriere_1, OUTPUT);
	pinMode(Moteurs_En_Arriere_2, OUTPUT);

	pinMode(Moteurs_Active_1, OUTPUT);
	pinMode(Moteurs_En_Avant_1, OUTPUT);
	pinMode(Moteurs_En_Avant_2, OUTPUT);
	
	// Commande des moteurs pour tourner à gauche
	digitalWrite(Moteurs_En_Arriere_1, HIGH);
	digitalWrite(Moteurs_En_Arriere_2, LOW);
	digitalWrite(Moteurs_En_Avant_1, LOW);
	digitalWrite(Moteurs_En_Avant_2, HIGH);
	analogWrite(Moteurs_Active_1, 255);  
	analogWrite(Moteurs_Active_2, 255);  
	
	delay(ROTATION_MOTEURS_TEMPS); 
	
	// Arrêter les moteurs
	analogWrite(Moteurs_Active_1, 0);
	analogWrite(Moteurs_Active_2, 0);
	
	return 7;
}

/**
 * @brief Arrête la voiture.
 * 
 * Coupe l’alimentation des moteurs pour immobiliser le véhicule.
 * 
 * @return Code de retour 8.
 */
int fonction_conduite_voiture_stop(void)
{
	// Configuration des broches
	pinMode(Moteurs_Active_1, OUTPUT);
	pinMode(Moteurs_En_Arriere_1, OUTPUT);
	pinMode(Moteurs_En_Arriere_2, OUTPUT);

	pinMode(Moteurs_Active_1, OUTPUT);
	pinMode(Moteurs_En_Avant_1, OUTPUT);
	pinMode(Moteurs_En_Avant_2, OUTPUT);
		
	// Arrêter les moteurs
	analogWrite(Moteurs_Active_1, 0);
	analogWrite(Moteurs_Active_2, 0);
	
	return 8;
}
