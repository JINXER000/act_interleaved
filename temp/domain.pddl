(define (domain open-world-tamp)
  (:requirements :strips :equality)
  (:constants
    @base @head @torso
  )
  (:predicates

    ; Types
    (Arm ?a)
    (Movable ?o)
    (Category ?o ?c)
    (Color ?o ?c)
    (ClosestColor ?o ?c)
    (Graspable ?o)
    (Controllable ?j)
    (Droppable ?o ?b)
    (Stackable ?o ?s)
    (Region ?s)
    (Reachable ?a ?o ?p ?bq)

    ; (CanPush ?o)
    (CanPick ?o)
    ; (CanPour ?o)
    ; (CanContain ?o)
    (CanMove ?a)

    (Pose ?o ?p)
    (InitPose ?o ?p)
    (Grasp ?a ?o ?g)
    (Conf ?j ?q)
    (RestConf ?j ?q)
    (Traj ?j ?t)
    (Material ?m)

    ; Static
    (Motion ?j ?q1 ?q2 ?bt)
    (Pick ?a ?o ?p ?g ?bq ?aq ?at)
    (Place ?a ?o ?p ?g ?bq ?aq ?at)
    (Supported ?o ?p ?s ?sp)

    (CFreePosePose ?o1 ?p1 ?o2 ?p2)
    (CFreePregraspPose ?a ?o1 ?p1 ?g1 ?o2 ?p2)
    (CFreeTrajPose ?j ?t ?o2 ?p2)

    (PoseLeftOf ?o1 ?p1 ?o2 ?p2)
    (PoseRightOf ?o1 ?p1 ?o2 ?p2)
    (PoseAheadOf ?o1 ?p1 ?o2 ?p2)
    (PoseBehind ?o1 ?p1 ?o2 ?p2)



    ; Fluent
    (AtConf ?j ?q)
    (AtPose ?o ?p)
    (AtGrasp ?a ?o ?g)
    (ArmEmpty ?a)
    (Localized ?o) ; Registered
   (ConfidentInPose ?o ?p)
    (HasPicked ?o)

    ; Derived
    (Resting ?j)
    (OtherActive ?j)
    (ArmHolding ?a ?o)
    (Holding ?o)
    (On ?o ?s)
    (Supporting ?s)

    (LeftOf ?o1 ?o2)
    (RightOf ?o1 ?o2)
    (AheadOf ?o1 ?o2)
    (Behind ?o1 ?o2)

    (UnsafePose ?o ?p)
    (UnsafePregrasp ?a ?o ?p ?g)
    (UnsafeTraj ?j ?t)


    ; for binamual insertion
    (PrecondConf ?a ?lc)
    (PrecondGrasp ?arm1 ?o1 ?lg1) 
    (EffGrasp ?arm1 ?o1 ?lg1) 
    (GeomState ?sstate)
    (DoneBiOp ?o1 ?o2)

    (peg ?o)
    (socket ?o)
    (left_arm ?arm)
    (right_arm ?arm)
    (CFreeBiOp ?arm1 ?arm2 ?q1 ?q2  ?o1 ?g1 ?o2 ?g2 ?o ?p)
    (UnsafeBiOp ?arm1 ?arm2 ?q1 ?q2 ?o1 ?g1 ?o2 ?g2)

    (AtKP ?j)
    (OtherNotAtKP ?j)
  )
  (:functions
    (MoveCost ?j)
    (PoseCost ?o ?p)
    (PlaceCost ?o ?s)
  )

  ;--------------------------------------------------

  ; TODO: increase cost if need to replan
  (:action move
    :parameters (?j ?q1 ?q2 ?t)
    :precondition (and (Motion ?j ?q1 ?q2 ?t)
                       ; (not (= ?q1 ?q2)) ; TODO: can't dop this with CanMove!
                       (CanMove ?j) ; TODO: account for multiple arms to ensure no deadlock
                       (or (not (OtherActive ?j)) (not (OtherNotAtKP ?j)))
                       
                       (AtConf ?j ?q1))
    :effect (and (AtConf ?j ?q2)
                 (not (AtConf ?j ?q1))
                 (not (CanMove ?j))
                 (increase (total-cost) (MoveCost ?j))))


  (:action pick
    :parameters (?a ?g ?o ?p ?bq ?aq ?at)
    :precondition (and (CanPick ?o)
                       (not (Supporting ?o))
                       (Pick ?a ?o ?p ?g ?bq ?aq ?at)
                       (AtPose ?o ?p) 
                       (ArmEmpty ?a) 
                       (AtConf ?a ?aq) 
                       (Reachable ?a ?o ?p ?bq)
                      ;  (Conf ?a ?aq) 
                       (AtConf @base ?bq)
                       (not (UnsafePregrasp ?a ?o ?p ?g))
                       (not (UnsafeTraj ?a ?at))
                  )
    :effect (and (AtGrasp ?a ?o ?g) (CanMove ?a) ; (AtConf ?a ?conf2)
                 (ArmHolding ?a ?o) (Holding ?o)
                 (HasPicked ?o) ; for testing grasp success
                 (not (AtPose ?o ?p)) (not (ArmEmpty ?a))
                 (not (ConfidentInPose ?o ?p))
                 (increase (total-cost) (PoseCost ?o ?p)))) 
  
  (:action place ; TODO: pick and drop action for testing grasp success
    :parameters (?a ?g ?o ?p ?s ?sp ?bq ?aq ?at)
    :precondition (and (Place ?a ?o ?p ?g ?bq ?aq ?at) (Supported ?o ?p ?s ?sp)
                        (Reachable ?a ?o ?p ?bq)
                       (AtGrasp ?a ?o ?g) (AtPose ?s ?sp) (AtConf ?a ?aq) (AtConf @base ?bq)
                       (not (UnsafePose ?o ?p))
                       (not (UnsafePregrasp ?a ?o ?p ?g))
                       (not (UnsafeTraj ?a ?at))
                  )
    :effect (and (AtPose ?o ?p) (ArmEmpty ?a) (CanMove ?a)
                 (not (AtGrasp ?a ?o ?g)) (not (Localized ?o))
                 (not (ArmHolding ?a ?o)) (not (Holding ?o))
                 (increase (total-cost) (PlaceCost ?o ?s))))


  ;--------------------------------------------------

  ; Derived predicates
  (:derived (Resting ?j)
    (exists (?q) (and (RestConf ?j ?q) ; (Conf ?j ?q)
                      (AtConf ?j ?q))))
  (:derived (OtherActive ?j)
    (exists (?a) (and (Arm ?a) (not (= ?j ?a))
                      (not (Resting ?a)))))

  (:derived (AtKP ?j)
    (exists (?q) (and (PrecondConf ?j ?q)
                      (AtConf ?j ?q))))

  (:derived (OtherNotAtKP ?j)
    (exists (?a) (and (Arm ?a) (not (= ?j ?a))
                      (not (AtKP ?a)))))
            

  (:derived (On ?o ?s)
    (exists (?p ?sp) (and (Supported ?o ?p ?s ?sp)
                          (AtPose ?o ?p)))
  )

  (:derived (LeftOf ?o1 ?o2)
    (exists (?p1 ?p2) (and (AtPose ?o1 ?p1) 
                           (AtPose ?o2 ?p2)
                           (PoseLeftOf ?o1 ?p1 ?o2 ?p2)))
  )
  (:derived (RightOf ?o1 ?o2)
    (exists (?p1 ?p2) (and (AtPose ?o1 ?p1) 
                           (AtPose ?o2 ?p2)
                           (PoseRightOf ?o1 ?p1 ?o2 ?p2)))
  )
  (:derived (AheadOf ?o1 ?o2)
    (exists (?p1 ?p2) (and (AtPose ?o1 ?p1) 
                           (AtPose ?o2 ?p2)
                           (PoseAheadOf ?o1 ?p1 ?o2 ?p2)))
  )
  (:derived (Behind ?o1 ?o2)
    (exists (?p1 ?p2) (and (AtPose ?o1 ?p1) 
                           (AtPose ?o2 ?p2)
                           (PoseBehind ?o1 ?p1 ?o2 ?p2)))
  )

  (:derived (Supporting ?s)
    (exists (?p ?sp ?o) (and (Supported ?o ?p ?s ?sp)
                             (AtPose ?o ?p)))
  )

    (:derived (On ?o ?s)
    (exists (?p) (and (Supported ?o ?p ?s ?sp)
                      (AtPose ?o ?p)))
  )

  (:derived (UnsafePose ?o1 ?p1) (and (Pose ?o1 ?p1)
    (exists (?o2 ?p2) (and (Pose ?o2 ?p2) (not (= ?o1 ?o2)) (Movable ?o2)
                           (not (CFreePosePose ?o1 ?p1 ?o2 ?p2))
                           (AtPose ?o2 ?p2)))))
  
  (:derived (UnsafePregrasp ?a ?o1 ?p1 ?g1) (and (Pose ?o1 ?o1) (Grasp ?a ?o1 ?g1)
    (exists (?o2 ?p2) (and (Pose ?o2 ?p2) (not (= ?o1 ?o2)) (Movable ?o2)
                           (not (CFreePregraspPose ?a ?o1 ?p1 ?g1 ?o2 ?p2))
                           (AtPose ?o2 ?p2)))))

  (:derived (UnsafeTraj ?j ?t) (and (Traj ?j ?t)
    (exists (?o2 ?p2) (and (Pose ?o2 ?p2) (Movable ?o2)
                           (not (CFreeTrajPose ?j ?t ?o2 ?p2))
                           (AtPose ?o2 ?p2)))))


  (:action BiOperation
    :parameters (?o1 ?o2 ?arm1 ?arm2 ?g1 ?g2 ?q1 ?q2 ); ?lt)
    :precondition (and (left_arm ?arm1) (right_arm ?arm2) 
                        (socket ?o1) (peg ?o2)

                        (PrecondConf ?arm1 ?q1) (PrecondConf ?arm2 ?q2)
                         (AtConf ?arm1 ?q1) (AtConf ?arm2 ?q2)

                        (PrecondGrasp ?arm1 ?o1 ?g1) (PrecondGrasp ?arm2 ?o2 ?g2) 
                        (AtGrasp ?arm1 ?o1 ?g1) (AtGrasp ?arm2 ?o2 ?g2) 
                        
                        (not (UnsafeBiOp ?arm1 ?arm2 ?q1 ?q2 ?o1 ?g1 ?o2 ?g2)) 
                  )
    ; NOTE: derived predicates cannot be used in the effect (https://baldur.iti.kit.edu/plan/files/getting-started-with-planning.pdf)
    :effect (and (DoneBiOp ?o1 ?o2) (CanMove ?arm1) (CanMove ?arm2)
                  (AtGrasp ?arm1 ?o1 ?g1) ; o2 attached to o1
                  (not (AtGrasp ?arm2 ?o2 ?g2))
                  ; AtLatent(?q1 ?q2)
            )
  )

  (:derived (UnsafeBiOp ?arm1 ?arm2 ?q1 ?q2 ?o1 ?g1 ?o2 ?g2)
    (exists (?o ?p) (and (Arm ?arm1) (Arm ?arm2) (not (= ?arm1 ?arm2))
                        (AtConf ?arm1 ?q1) (AtConf ?arm2 ?q2)
                        (AtGrasp ?arm1 ?o1 ?g1) (AtGrasp ?arm2 ?o2 ?g2)
                        (not (CFreeBiOp ?arm1 ?arm2 ?q1 ?q2  ?o1 ?g1 ?o2 ?g2 ?o ?p))
                        (AtPose ?o ?p) (Movable ?o) (CanPick ?o) (not (Holding ?o))
                    )
    )
  )


)