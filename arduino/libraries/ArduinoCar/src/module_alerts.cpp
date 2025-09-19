/// @file module_alerts.cpp
/// @brief Implémentation des fonctions d’alerte pour la voiture.
/// 
/// Ce module contrôle le klaxon (buzzer) et les LED RGB pour simuler
/// les phares ou les clignotants de la voiture connectée.

#include "Arduino.h"
#include "module_alerts.hpp"

/// @brief Broche utilisée pour le buzzer (klaxon).
const int buzzerPin = 8;

/// @brief Broche pour la LED bleue (RGB).
const int rgbLEDbleu = 4;

/// @brief Broche pour la LED verte (RGB).
const int rgbLEDvert = 5;

/// @brief Broche pour la LED rouge (RGB).
const int rgbLEDrouge = 6;

/**
 * @brief Joue un son de klaxon via le buzzer.
 * 
 * Émet deux sons consécutifs simulant un klaxon de voiture.
 */
void playHorn() {
  // Fréquence typique d'un klaxon de voiture : ~440 à 500 Hz
  tone(buzzerPin, 460);      // Active le buzzer à 460 Hz
  delay(500);                // Durée du son
  noTone(buzzerPin);         // Arrête le son
  delay(200);                // Pause entre deux signaux
  tone(buzzerPin, 460);      // Rejoue le son
  delay(500);
  noTone(buzzerPin);
}

/**
 * @brief Active le klaxon de la voiture.
 * 
 * Configure la broche du buzzer et déclenche la fonction de klaxon.
 * 
 * @return 1 pour indiquer le succès.
 */
int fonction_alertes_voiture_klaxonne(void)
{
	pinMode(buzzerPin, OUTPUT);
	playHorn();
	return 1;
}

/**
 * @brief Allume les pleins phares (LED blanche).
 * 
 * Active les trois canaux de la LED RGB pour afficher du blanc.
 * 
 * @return 2 pour indiquer le succès.
 */
int fonction_alertes_voiture_plein_phare(void)
{
	pinMode(rgbLEDbleu, OUTPUT);
	pinMode(rgbLEDvert, OUTPUT);
	pinMode(rgbLEDrouge, OUTPUT);

	// Allumer le blanc (R + G + B à fond - anode commune)
	analogWrite(rgbLEDbleu, 0);
	analogWrite(rgbLEDvert, 0);
	analogWrite(rgbLEDrouge, 0);

	return 2;
}

/**
 * @brief Fait clignoter les phares avant.
 * 
 * Allume et éteint la LED RGB blanche trois fois pour simuler un clignotement.
 * 
 * @return 3 pour indiquer le succès.
 */
int fonction_alertes_voiture_clignotte(void)
{
	pinMode(rgbLEDbleu, OUTPUT);
	pinMode(rgbLEDvert, OUTPUT);
	pinMode(rgbLEDrouge, OUTPUT);

	unsigned int blink_counter = 3;
	
	for(unsigned int blink = 0; blink < blink_counter; blink++)
	{
		// Allume la LED RGB en blanc
		analogWrite(rgbLEDbleu, 0);
		analogWrite(rgbLEDvert, 0);
		analogWrite(rgbLEDrouge, 0);
		
		delay(500);

		// Éteint la LED RGB
		analogWrite(rgbLEDbleu, 255);
		analogWrite(rgbLEDvert, 255);
		analogWrite(rgbLEDrouge, 255);

		delay(500);		
	}
		
	return 3;
}
