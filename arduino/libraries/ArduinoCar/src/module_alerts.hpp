/// @file module_alerts.hpp
/// @brief Déclarations des fonctions d'alerte pour la voiture.
/// 
/// Ce module fournit les fonctions associées aux systèmes d'alerte de la voiture,
/// telles que le klaxon, les phares et les clignotants.

/**
 * @brief Déclenche le klaxon de la voiture.
 * 
 * @return 0 en cas de succès, autre valeur en cas d'erreur.
 */
int fonction_alertes_voiture_klaxonne(void);

/**
 * @brief Active les pleins phares de la voiture.
 * 
 * @return 0 en cas de succès, autre valeur en cas d'erreur.
 */
int fonction_alertes_voiture_plein_phare(void);

/**
 * @brief Active les clignotants de la voiture.
 * 
 * @return 0 en cas de succès, autre valeur en cas d'erreur.
 */
int fonction_alertes_voiture_clignotte(void);
