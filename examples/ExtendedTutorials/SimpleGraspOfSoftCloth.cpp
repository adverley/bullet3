//
// Created by averleysen on 2/20/17.
//

#include <BulletSoftBody/btSoftRigidDynamicsWorld.h>
#include <BulletSoftBody/btSoftBodyRigidBodyCollisionConfiguration.h>
#include "SimpleGraspOfSoftCloth.h"
#include "../CommonInterfaces/CommonExampleInterface.h"
#include "../CommonInterfaces/CommonRigidBodyBase.h"
#include "../CommonInterfaces/CommonParameterInterface.h"
#include "../RoboticsLearning/b3RobotSimAPI.h"

static btScalar sGripperVerticalVelocity = 0.f;
static btScalar sGripperClosingTargetVelocity = -0.7f;

struct SimpleGraspOfSoftCloth : public CommonRigidBodyBase {
    CommonGraphicsApp *m_app;
    b3RobotSimAPI m_robotSim;
    int m_options;
    int m_gripperIndex;
    btSoftBodyWorldInfo softBodyWorldInfo;

    SimpleGraspOfSoftCloth(struct GUIHelperInterface *helper, int options = 0) : CommonRigidBodyBase(helper) {
        m_options = options;
        m_gripperIndex = -1;
    }

    virtual ~SimpleGraspOfSoftCloth() {
//        m_app->m_renderer->enableBlend(false);
    }

    virtual void initPhysics();

    virtual void exitPhysics();

    virtual void stepSimulation(float deltaTime);

    virtual void renderScene();

    void resetCamera() {
        float dist = 1;
        float pitch = 52;
        float yaw = 35;
        float targetPos[3] = {0, 0, 0};
        m_guiHelper->resetCamera(dist, pitch, yaw, targetPos[0], targetPos[1], targetPos[2]);
    }

    void createGround();

    void createObstructingBox();

    void createWorld();

    void createGripper();

    void createCloth();

    void add_slider_controls() const;

    void addJointBetweenFingerAndMotor();

    virtual btSoftRigidDynamicsWorld *getSoftDynamicsWorld() {
        ///just make it a btSoftRigidDynamicsWorld please
        return (btSoftRigidDynamicsWorld *) m_dynamicsWorld;
    }

    void doMotorControl();
};

void SimpleGraspOfSoftCloth::initPhysics() {
    bool connected = m_robotSim.connect(m_guiHelper);
    b3Printf("robotSim connected = %d", connected);

    createWorld();
    createGround();
//    createObstructingBox();
    createGripper();
    createCloth();
    m_guiHelper->autogenerateGraphicsObjects(m_dynamicsWorld);
}

void SimpleGraspOfSoftCloth::createWorld() {
    m_guiHelper->setUpAxis(1);

///    setupCollisionDetection
    /* 3D collision detection can become very complex, especially when using specialized algorithms.
     * That is why we work in two phases:
     *  1. Find collision of objects which are close to each other -> broad phase
     *  2. Use a more accurate collision algorithm for those objects close to each other -> collision dispatcher
     *  */

    /* The collision configuration specifies how memory will be allocated and managed for collision detection */
    m_collisionConfiguration = new btSoftBodyRigidBodyCollisionConfiguration();
    /*  the dispatcher calculates collision of objects close to each other accurately */
    m_dispatcher = new btCollisionDispatcher(m_collisionConfiguration);
    /* The broadphase checks which objects collides and which don't for a quick raw view of colliding objects */
    m_broadphase = new btDbvtBroadphase();

    /* The constraint solver will attach objects to each other */
    m_solver = new btSequentialImpulseConstraintSolver;

    /* Throw all the ingredients for collision detection in soft body world together */
    m_dynamicsWorld = new btSoftRigidDynamicsWorld(m_dispatcher, m_broadphase, m_solver, m_collisionConfiguration);
    m_dynamicsWorld->setGravity(btVector3(0, -10, 0));

    /* Hold some bookkeeping about the soft body world */
    softBodyWorldInfo.m_broadphase = m_broadphase;
    softBodyWorldInfo.m_dispatcher = m_dispatcher;
    softBodyWorldInfo.m_gravity = m_dynamicsWorld->getGravity();
    softBodyWorldInfo.m_sparsesdf.Initialize();

    m_guiHelper->createPhysicsDebugDrawer(m_dynamicsWorld);

    if (m_dynamicsWorld->getDebugDrawer())
        m_dynamicsWorld->getDebugDrawer()->setDebugMode(
                btIDebugDraw::DBG_DrawWireframe + btIDebugDraw::DBG_DrawContactPoints);
    m_robotSim.debugDraw(true);
}


void SimpleGraspOfSoftCloth::createGround() {
    btBoxShape *groundShape = createBoxShape(btVector3(btScalar(50.), btScalar(50.), btScalar(50.)));
    m_collisionShapes.push_back(groundShape);

    btTransform groundTransform;
    groundTransform.setIdentity();
    groundTransform.setOrigin(btVector3(0, -50, 0));

    {
        btScalar mass(0.);
        createRigidBody(mass, groundTransform, groundShape, btVector4(0, 0, 1, 1));
    }

}

void SimpleGraspOfSoftCloth::createObstructingBox() {
    btBoxShape *colShape = createBoxShape(btVector3(.1, .1, .1));
    m_collisionShapes.push_back(colShape);

    btTransform startTransform;
    startTransform.setIdentity();

    btScalar mass(1.f);
    bool isDynamic = (mass != 0.f);

    btVector3 localInertia(0, 0, 0);
    if (isDynamic)
        colShape->calculateLocalInertia(mass, localInertia);

    startTransform.setOrigin(btVector3(0, 0, 0));
    createRigidBody(mass, startTransform, colShape);
}

void SimpleGraspOfSoftCloth::createGripper() {
    m_robotSim.setNumSolverIterations(150);
    add_slider_controls();
    {
        b3RobotSimLoadFileArgs args("");
        args.m_fileName = "gripper/wsg50_one_motor_gripper_new.sdf";
        args.m_fileType = B3_SDF_FILE;
        args.m_useMultiBody = true;
        args.m_startPosition.setValue(0, 0, 0);
        args.m_startOrientation.setEulerZYX(SIMD_PI, SIMD_HALF_PI, 0);
        b3RobotSimLoadFileResults results;

        if (m_robotSim.loadFile(args, results) && results.m_uniqueObjectIds.size() == 1) {

            b3Printf("GRIPPER INFORMATION:");
            m_gripperIndex = results.m_uniqueObjectIds[0];
            int numJoints = m_robotSim.getNumJoints(m_gripperIndex);
            b3Printf("numJoints = %d", numJoints);

            for (int i = 0; i < numJoints; i++) {
                b3JointInfo jointInfo;
                m_robotSim.getJointInfo(m_gripperIndex, i, &jointInfo);
                b3Printf("joint[%d].m_jointName=%s", i, jointInfo.m_jointName);
            }

            for (int i = 0; i < 8; i++) {
                b3JointMotorArgs controlArgs(CONTROL_MODE_VELOCITY);
                controlArgs.m_maxTorqueValue = 0.0;
                m_robotSim.setJointMotorControl(m_gripperIndex, i, controlArgs);
            }

        }
        addJointBetweenFingerAndMotor();
    }

}

void SimpleGraspOfSoftCloth::addJointBetweenFingerAndMotor() {
    b3JointInfo revoluteJoint1;
    revoluteJoint1.m_parentFrame[0] = -0.055;
    revoluteJoint1.m_parentFrame[1] = 0;
    revoluteJoint1.m_parentFrame[2] = 0.02;
    revoluteJoint1.m_parentFrame[3] = 0;
    revoluteJoint1.m_parentFrame[4] = 0;
    revoluteJoint1.m_parentFrame[5] = 0;
    revoluteJoint1.m_parentFrame[6] = 1.0;
    revoluteJoint1.m_childFrame[0] = 0;
    revoluteJoint1.m_childFrame[1] = 0;
    revoluteJoint1.m_childFrame[2] = 0;
    revoluteJoint1.m_childFrame[3] = 0;
    revoluteJoint1.m_childFrame[4] = 0;
    revoluteJoint1.m_childFrame[5] = 0;
    revoluteJoint1.m_childFrame[6] = 1.0;
    revoluteJoint1.m_jointAxis[0] = 1.0;
    revoluteJoint1.m_jointAxis[1] = 0.0;
    revoluteJoint1.m_jointAxis[2] = 0.0;
    revoluteJoint1.m_jointType = ePoint2PointType;

    b3JointInfo revoluteJoint2;
    revoluteJoint2.m_parentFrame[0] = 0.055;
    revoluteJoint2.m_parentFrame[1] = 0;
    revoluteJoint2.m_parentFrame[2] = 0.02;
    revoluteJoint2.m_parentFrame[3] = 0;
    revoluteJoint2.m_parentFrame[4] = 0;
    revoluteJoint2.m_parentFrame[5] = 0;
    revoluteJoint2.m_parentFrame[6] = 1.0;
    revoluteJoint2.m_childFrame[0] = 0;
    revoluteJoint2.m_childFrame[1] = 0;
    revoluteJoint2.m_childFrame[2] = 0;
    revoluteJoint2.m_childFrame[3] = 0;
    revoluteJoint2.m_childFrame[4] = 0;
    revoluteJoint2.m_childFrame[5] = 0;
    revoluteJoint2.m_childFrame[6] = 1.0;
    revoluteJoint2.m_jointAxis[0] = 1.0;
    revoluteJoint2.m_jointAxis[1] = 0.0;
    revoluteJoint2.m_jointAxis[2] = 0.0;
    revoluteJoint2.m_jointType = ePoint2PointType;

    int bodyIndex = 0;
    int motor_left_hinge_joint = 2;
    int motor_right_hinge_joint = 2;
    int gripper_left_hinge_joint = 4;
    int gripper_right_hinge_joint = 6;
    m_robotSim.createJoint(bodyIndex, motor_left_hinge_joint, bodyIndex, gripper_left_hinge_joint, &revoluteJoint1);
    m_robotSim.createJoint(bodyIndex, motor_right_hinge_joint, bodyIndex, gripper_right_hinge_joint, &revoluteJoint2);
}

void SimpleGraspOfSoftCloth::add_slider_controls() const {
    {
        SliderParams slider("Vertical velocity", &sGripperVerticalVelocity);
        slider.m_minVal = -2;
        slider.m_maxVal = 2;
        m_guiHelper->getParameterInterface()->registerSliderFloatParameter(slider);
    }

    {
        SliderParams slider("Closing velocity", &sGripperClosingTargetVelocity);
        slider.m_minVal = -1;
        slider.m_maxVal = 1;
        m_guiHelper->getParameterInterface()->registerSliderFloatParameter(slider);
    }
}

void SimpleGraspOfSoftCloth::createCloth() {
// TODO see extended tutoirals -> simple cloth cpp
//    also see experiments/grasp soft body
}

void SimpleGraspOfSoftCloth::renderScene() {
    CommonRigidBodyBase::renderScene();
}

void SimpleGraspOfSoftCloth::exitPhysics() {
    m_robotSim.disconnect();
}

void SimpleGraspOfSoftCloth::stepSimulation(float deltaTime) {
    doMotorControl();

    m_robotSim.stepSimulation();
//    CommonRigidBodyBase::stepSimulation(deltaTime);

}

void SimpleGraspOfSoftCloth::doMotorControl() {
    int fingerJointIndices[2] = {0, 1};
    double fingerTargetVelocities[2] = {sGripperVerticalVelocity, sGripperClosingTargetVelocity};
    double maxTorqueValues[2] = {50.0, 10.0};
    for (int i = 0; i < 2; i++) {
        b3JointMotorArgs controlArgs(CONTROL_MODE_VELOCITY);
        controlArgs.m_targetVelocity = fingerTargetVelocities[i];
        controlArgs.m_maxTorqueValue = maxTorqueValues[i];
        controlArgs.m_kd = 1.;
        m_robotSim.setJointMotorControl(m_gripperIndex, fingerJointIndices[i], controlArgs);
    }
}

CommonExampleInterface *ET_SimpleGraspOfSoftClothCreateFunc(CommonExampleOptions &options) {
    return new SimpleGraspOfSoftCloth(options.m_guiHelper);
}
