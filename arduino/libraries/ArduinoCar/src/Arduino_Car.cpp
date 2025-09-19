/// @file Arduino_Car.cpp
/// @brief Implémentation des méthodes de la classe Arduino_Car.
/// 
/// Ce fichier relie les méthodes de haut niveau de la classe Arduino_Car
/// aux fonctions des modules d'alerte et de conduite, et stocke l'action
/// exécutée dans l'attribut `action_car`.

#include "Arduino_Car.hpp"

/*	Méthodes	*/

/**
 * @brief Constructeur de la classe Arduino_Car.
 * 
 * Initialise l'attribut action_car avec une valeur par défaut (par exemple -1 pour signifier aucune action).
 */
Arduino_Car::Arduino_Car()
{
	action_car = -1;  // Valeur par défaut indiquant "aucune action"
}

/**
 * @brief Destructeur de la classe Arduino_Car.
 * 
 * Arrête la voiture pour s'assurer d'une extinction propre.
 */
Arduino_Car::~Arduino_Car()
{
	action_car = fonction_conduite_voiture_stop();  // Par sécurité, on arrête la voiture à la destruction de l'objet
}

/**
 * @brief Fait avancer la voiture.
 * 
 * Appelle la fonction de conduite correspondante et stocke le code d'action.
 */
void Arduino_Car::avance(void)
{
	action_car = fonction_conduite_voiture_avance();
}

/**
 * @brief Fait reculer la voiture.
 * 
 * Appelle la fonction de conduite correspondante et stocke le code d'action.
 */
void Arduino_Car::recule(void)
{
	action_car = fonction_conduite_voiture_recule();
}

/**
 * @brief Fait tourner la voiture à gauche.
 * 
 * Appelle la fonction de conduite correspondante et stocke le code d'action.
 */
void Arduino_Car::tourne_a_gauche(void)
{
	action_car = fonction_conduite_voiture_tourne_a_gauche();
}

/**
 * @brief Fait tourner la voiture à droite.
 * 
 * Appelle la fonction de conduite correspondante et stocke le code d'action.
 */
void Arduino_Car::tourne_a_droite(void)
{
	action_car = fonction_conduite_voiture_tourne_a_droite();
}

/**
 * @brief Arrête la voiture.
 * 
 * Appelle la fonction de conduite correspondante et stocke le code d'action.
 */
void Arduino_Car::stop(void)
{
	action_car = fonction_conduite_voiture_stop();
}

/**
 * @brief Active le klaxon de la voiture.
 * 
 * Appelle la fonction d'alerte correspondante et stocke le code d'action.
 */
void Arduino_Car::klaxonne(void)
{
	action_car = fonction_alertes_voiture_klaxonne();
}

/**
 * @brief Fait clignoter les feux de la voiture.
 * 
 * Appelle la fonction d'alerte correspondante et stocke le code d'action.
 */
void Arduino_Car::clignotte(void)
{
	action_car = fonction_alertes_voiture_clignotte();
}

/**
 * @brief Allume les pleins phares de la voiture.
 * 
 * Appelle la fonction d'alerte correspondante et stocke le code d'action.
 */
void Arduino_Car::plein_phare(void)
{
	action_car = fonction_alertes_voiture_plein_phare();
}
