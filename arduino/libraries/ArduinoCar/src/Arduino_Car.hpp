/// @file Arduino_Car.hpp
/// @brief Déclaration de la classe Arduino_Car gérant les actions de la voiture.
/// 
/// Cette classe centralise les actions de conduite et d'alerte de la voiture connectée.
/// Elle utilise les modules Bluetooth, d'alertes et de conduite.

#include "module_alerts.hpp"
#include "module_driving.hpp"

/**
 * @class Arduino_Car
 * @brief Classe représentant les fonctionnalités de la voiture.
 * 
 * Fournit des méthodes pour contrôler la voiture : avancer, reculer, tourner,
 * ainsi que pour déclencher les alertes comme le klaxon, les clignotants et les phares.
 */
class Arduino_Car
{
	public :
	
	/*	Constructeur	*/
	Arduino_Car();
	
	/*	Destructeur	*/
	~Arduino_Car();

	/*	Attributs	*/

	/// @brief Dernière action effectuée par la voiture (code d'action).
	int action_car;
	
	/*	Méthodes	*/

	/// @brief Fait avancer la voiture.
	void avance(void);
	
	/// @brief Fait reculer la voiture.
	void recule(void);
	
	/// @brief Fait tourner la voiture à gauche.
	void tourne_a_gauche(void);
	
	/// @brief Fait tourner la voiture à droite.
	void tourne_a_droite(void);
	
	/// @brief Arrête la voiture.
	void stop(void);
	
	/// @brief Active le klaxon.
	void klaxonne(void);
	
	/// @brief Active les clignotants.
	void clignotte(void);
	
	/// @brief Active les pleins phares.
	void plein_phare(void);
};
