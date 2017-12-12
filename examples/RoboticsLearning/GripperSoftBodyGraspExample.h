
#ifndef PROJECT_GRIPPERSOFTBODYGRASPEXAMPLE_H
#define PROJECT_GRIPPERSOFTBODYGRASPEXAMPLE_H

enum GripperSoftBodyGraspExampleOptions
{
    SOFT_BODY_GRASPING=1,
    SOFT_BODY_WITH_MULTIBODY_COUPLING=2,
};

class CommonExampleInterface*    GripperSoftBodyGraspExampleCreateFunc(struct CommonExampleOptions& options);

#endif //PROJECT_GRIPPERSOFTBODYGRASPEXAMPLE_H
