/// @file module_driving.hpp
/// @brief Déclarations des fonctions de conduite de la voiture.
/// 
/// Ce module fournit les fonctions permettant de contrôler les mouvements de la voiture
/// tels qu’avancer, reculer, tourner ou s’arrêter.

/**
 * @brief Fait avancer la voiture.
 * 
 * @return 0 en cas de succès, autre valeur en cas d'erreur.
 */
int fonction_conduite_voiture_avance(void);

/**
 * @brief Fait reculer la voiture.
 * 
 * @return 0 en cas de succès, autre valeur en cas d'erreur.
 */
int fonction_conduite_voiture_recule(void);

/**
 * @brief Fait tourner la voiture à gauche.
 * 
 * @return 0 en cas de succès, autre valeur en cas d'erreur.
 */
int fonction_conduite_voiture_tourne_a_gauche(void);

/**
 * @brief Fait tourner la voiture à droite.
 * 
 * @return 0 en cas de succès, autre valeur en cas d'erreur.
 */
int fonction_conduite_voiture_tourne_a_droite(void);

/**
 * @brief Arrête la voiture.
 * 
 * @return 0 en cas de succès, autre valeur en cas d'erreur.
 */
int fonction_conduite_voiture_stop(void);
