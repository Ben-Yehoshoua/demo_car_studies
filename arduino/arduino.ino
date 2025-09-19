/**
 * @file Arduino_Car.ino
 * @brief Projet Earth_Drone.
 * 
 *   Ce programme a été développé pour piloter une maquette de voiture.
 *   Il repose sur le système d'exploitation FreeRTOS, chargé de 
 *   séquencer les différentes tâches, telles que la communication 
 *   Bluetooth, la gestion de la conduite, ainsi que le contrôle des 
 *   alertes sonores et lumineuses.
 * 
 * @author Nicolas CHANGO
 * @date 2025-05-19
 * @version 1.0
 */

/************************************************************
 *                     INCLUSIONS                           *
 ************************************************************/
// Inclusion de bibliothèques standard ou spécifiques
#include <Arduino_FreeRTOS.h>       ///< Bibliothèque Arduino_FreeRTOS pour la gestion multitâche.
#include <queue.h>                  ///< Bibliothèque queue pour la gestion des files.
#include <SoftwareSerial.h>         ///< Bibliothèque SoftwareSerial pour la communication série.
#include <Arduino_Car.hpp>  ///< Header Arduino_car.hpp pour la description de la classe Arduino_Car

/************************************************************
 *                  DÉFINITIONS ET CONSTANTES               *
 ************************************************************/
/// @brief Active le mode debug (1 = activé, 0 = désactivé)
#define DEBUG_MODE (1u)

/// @brief Broche utilisée pour recevoir les données TX du module Bluetooth HC-06.
#define BLUETOOTH_RX  (2U)

/// @brief Broche utilisée pour recevoir les données RX du module Bluetooth HC-06.
#define BLUETOOTH_TX  (3U)

/// @brief Taille maximale des messages dans les files d'alertes et de conduite.
#define MAX_STRING_LENGTH (100U)

/// @brief File de messages pour les alertes.
QueueHandle_t Alerts_Queue;

/// @brief File de messages pour les commandes de conduite.
QueueHandle_t Driving_Queue;

/// @brief Objet Serial pour la communication Bluetooth avec le module HC-06.
SoftwareSerial BLUETOOTH_MODULE_HC06(BLUETOOTH_RX, BLUETOOTH_TX);

/// @brief Création de l'objet Earth_Drone.
Arduino_Car Earth_Drone;

/************************************************************
 *                    PROTOTYPES DE FONCTIONS               *
 ************************************************************/
/// @brief Tâche de gestion Bluetooth
void Task_Bluetooth( void *pvParameters );

/// @brief Tâche de gestion des alertes
void Task_Alerts( void *pvParameters );

/// @brief Tâche de gestion des commandes de conduite
void Task_Driving( void *pvParameters );

/************************************************************
 *                    CONFIGURATION INITIALE (setup)        *
 ************************************************************/
/**
 * @brief Fonction d'initialisation Arduino.
 * 
 * Initialise les communications série, les files de messages FreeRTOS
 * et crée les différentes tâches (Bluetooth, Alerts, Driving) si la
 * création des files est réussie.
 */
void setup() 
{ 
  /// Initialise la communication avec le module Bluetooth HC-06 à 9600 bauds
  BLUETOOTH_MODULE_HC06.begin(9600);

  /// Initialise le port série standard à 9600 bauds
  Serial.begin(9600);

  /// Création de la file de messages pour les alertes (taille 10, type String)
  Alerts_Queue = xQueueCreate(10, 
                              sizeof(String));

  /// Création de la file de messages pour la conduite (taille 10, type String)
  Driving_Queue = xQueueCreate(10, 
                                  sizeof(String)); 
  
  /// Vérifie que les deux files ont été créées avec succès
  if ((Alerts_Queue != NULL) && (Driving_Queue != NULL))
  {
    /**
     * @brief Création de la tâche Bluetooth.
     * 
     * Priorité : 2, Pile : 128 mots.
     */
    xTaskCreate(
      Task_Bluetooth,
      "Bluetooth",   // Nom de la tâche (à des fins de debug)
      128,           // Taille de la pile
      NULL,          // Paramètres (aucun ici)
      2,             // Priorité
      NULL           // Handle de tâche non utilisé
    );

    /**
     * @brief Création de la tâche Alerts.
     * 
     * Priorité : 1, Pile : 128 mots.
     */
    xTaskCreate(
      Task_Alerts,
      "Alerts",
      128,
      NULL,
      1,
      NULL
    );

    /**
     * @brief Création de la tâche Driving.
     * 
     * Priorité : 0, Pile : 128 mots.
     */
    xTaskCreate(
      Task_Driving,
      "Driving",
      128,
      NULL,
      0,
      NULL
    );
  }
}

/************************************************************
 *                  BOUCLE PRINCIPALE (loop)                *
 ************************************************************/
/**
 * @brief Boucle principale Arduino.
 * 
 * Dans le contexte d'une application FreeRTOS, cette boucle peut rester vide
 * ou servir à surveiller des éléments non critiques.
 */
void loop()
{
  // Optionnel ou vide si FreeRTOS gère tout
}

/************************************************************
 *                 DÉFINITIONS DES TÂCHES                   *
 ************************************************************/
/**
 * @brief Tâche pour gérer les communications Bluetooth.
 * 
 * Cette tâche traite les messages reçus par Bluetooth.
 * Elle les renvoie via des "files" aux autres tâches.
 * Elle s'exécute indéfiniment dans une boucle FreeRTOS.
 * 
 * @param pvParameters Paramètre passé à la tâche (non utilisé ici).
 */
/**
 * @brief Tâche pour recevoir les messages Bluetooth.
 * 
 * Cette tâche lit les caractères reçus du module Bluetooth HC-06,
 * les assemble en un message complet, et les transmet dans deux files
 * FreeRTOS : Alerts_Queue et Driving_Queue.
 * 
 * @param pvParameters Paramètre passé à la tâche (non utilisé ici).
 */
void Task_Bluetooth(void *pvParameters)  
{
  (void) pvParameters;

  /// Caractère courant reçu via Bluetooth
  char Bluetooth_Caractere_Recu = '0';
  
  /// Message complet reçu via Bluetooth
  String Bluetooth_Message_Recu = "";

  for (;;) 
  {
    /// Réinitialisation du message à chaque itération
    Bluetooth_Message_Recu = "";
    
    /// Lecture des caractères disponibles sur le module Bluetooth
    while(BLUETOOTH_MODULE_HC06.available())
    {
      vTaskDelay(1);  ///< Laisse du temps aux autres tâches
      Bluetooth_Caractere_Recu = BLUETOOTH_MODULE_HC06.read();
      Bluetooth_Message_Recu += Bluetooth_Caractere_Recu;
    }

    /// Si un message a été reçu, le transmettre aux files
    if(Bluetooth_Message_Recu.length() > 0)
    {

#if DEBUG_MODE
      /// Affiche le message reçu dans le terminal série (mode debug)
      Serial.println(Bluetooth_Message_Recu);
#endif

      /// Envoie le message à la file d'alertes
      xQueueSend(Alerts_Queue, &Bluetooth_Message_Recu, portMAX_DELAY);

      /// Envoie le message à la file de conduite
      xQueueSend(Driving_Queue, &Bluetooth_Message_Recu, portMAX_DELAY);
    }

    vTaskDelay(1);  ///< Petite pause pour éviter la surcharge CPU
  }
}

/**
 * @brief Tâche pour gérer les Alertes sur la voiture.
 * 
 * Cette tâche récupère les messages envoyés par la tâche "Task_Bluetooth".
 * Elle appelle les fonctions d'alertes de la voiture selon le message Bluetooth reçu.
 * Elle s'exécute indéfiniment dans une boucle FreeRTOS.
 * 
 * @param pvParameters Paramètre passé à la tâche (non utilisé ici).
 */
void Task_Alerts(void *pvParameters)  
{
  (void) pvParameters;
  
  /// Message reçu depuis la file Alerts_Queue
  String Task_Bluetooth_Commande_Recu = "";

  for (;;) 
  {
    /// Récupère un message de la file d'alertes
    if(xQueueReceive(Alerts_Queue, &Task_Bluetooth_Commande_Recu, portMAX_DELAY) == pdPASS)
    {
      Task_Bluetooth_Commande_Recu.trim();

      /// Analyse du message reçu et appel de la fonction d'alerte correspondante
      if(!strcmp("klaxonne", Task_Bluetooth_Commande_Recu.c_str()))
      {
        Earth_Drone.klaxonne();
      }
      else if(!strcmp("plein_phare", Task_Bluetooth_Commande_Recu.c_str()))
      {
        Earth_Drone.plein_phare();
      }
      else if(!strcmp("clignotte", Task_Bluetooth_Commande_Recu.c_str()))
      {
        Earth_Drone.clignotte();
      }
      else
      {
        /* Rien à faire */
      }
    }
    else
    {
      /// Affiche une erreur si la réception de la file échoue
      Serial.println("Erreur");
    }

#if DEBUG_MODE        
    /// Affiche l'action exécutée dans le terminal série (mode debug)
    Serial.println(Earth_Drone.action_car);
#endif

    vTaskDelay(1);
  }
}

/**
 * @brief Tâche pour gérer la conduite de la voiture.
 * 
 * Cette tâche récupère les messages envoyés par la tâche "Task_Bluetooth".
 * Elle appelle les fonctions de conduite de la voiture selon le message Bluetooth reçu.
 * Elle s'exécute indéfiniment dans une boucle FreeRTOS.
 * 
 * @param pvParameters Paramètre passé à la tâche (non utilisé ici).
 */
void Task_Driving(void *pvParameters)  
{
  (void) pvParameters;
  
  /// Message reçu depuis la file Driving_Queue
  String Task_Bluetooth_Commande_Recu = "";

  for (;;) 
  {
    /// Récupère un message de la file de conduite
    if(xQueueReceive(Driving_Queue, &Task_Bluetooth_Commande_Recu, portMAX_DELAY) == pdPASS)
    {
      Task_Bluetooth_Commande_Recu.trim();

      /// Analyse du message reçu et appel de la fonction de conduite correspondante
      if(!strcmp("avance", Task_Bluetooth_Commande_Recu.c_str()))
      {
        Earth_Drone.avance();
      }
      else if(!strcmp("recule", Task_Bluetooth_Commande_Recu.c_str()))
      {
        Earth_Drone.recule();
      }
      else if(!strcmp("tourne_a_droite", Task_Bluetooth_Commande_Recu.c_str()))
      {
        Earth_Drone.tourne_a_droite();
      }
      else if(!strcmp("tourne_a_gauche", Task_Bluetooth_Commande_Recu.c_str()))
      {
        Earth_Drone.tourne_a_gauche();
      }
      else if(!strcmp("stop", Task_Bluetooth_Commande_Recu.c_str()))
      {
        Earth_Drone.stop();
      }
      else
      {
        /* Rien à faire */
      }
    }
    else
    {
      /// Affiche une erreur si la réception de la file échoue
      Serial.println("Erreur");
    }

#if DEBUG_MODE        
    /// Affiche l'action exécutée dans le terminal série (mode debug)
    Serial.println(Earth_Drone.action_car);
#endif

    vTaskDelay(1);
  }
}
