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

struct SimpleGraspOfSoftCloth : public CommonExampleInterface {
    CommonGraphicsApp *m_app;
    GUIHelperInterface *m_guiHelper;
    btDiscreteDynamicsWorld *m_dynamicsWorld;
    btAlignedObjectArray<btCollisionShape *> m_collisionShapes;
    btBroadphaseInterface *m_broadphase;
    btCollisionDispatcher *m_dispatcher;
    btConstraintSolver *m_solver;
    btDefaultCollisionConfiguration *m_collisionConfiguration;

    b3RobotSimAPI m_robotSim;
    int m_options;
    int m_gripperIndex;
    btSoftBodyWorldInfo softBodyWorldInfo;

    SimpleGraspOfSoftCloth(struct GUIHelperInterface *helper, int options = 0) {
        m_gripperIndex = -1;
        m_app = helper->getAppInterface();
        m_guiHelper = helper;
        m_options = options;
    }

    virtual ~SimpleGraspOfSoftCloth() {
        m_app->m_renderer->enableBlend(false);
    }

    virtual void physicsDebugDraw(int debugFlags) {

    }

    virtual bool mouseMoveCallback(float x, float y) {
        return false;
    }

    virtual bool mouseButtonCallback(int button, int state, float x, float y) {
        return false;
    }

    virtual bool keyboardCallback(int key, int state) {
        return false;
    }

    virtual void initPhysics();

    virtual void exitPhysics();

    virtual void renderScene();

    void resetCamera() {
        float dist = 1;
        float pitch = 52;
        float yaw = 35;
        float targetPos[3] = {0, 0, 0};
        if (m_app->m_renderer && m_app->m_renderer->getActiveCamera()) {
            m_app->m_renderer->getActiveCamera()->setCameraDistance(dist);
            m_app->m_renderer->getActiveCamera()->setCameraPitch(pitch);
            m_app->m_renderer->getActiveCamera()->setCameraYaw(yaw);
            m_app->m_renderer->getActiveCamera()->setCameraTargetPosition(targetPos[0], targetPos[1], targetPos[2]);
        }
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

    virtual void stepSimulation(float deltaTime);

    btRigidBody *createRigidBody(btScalar mass, btTransform transform, btBoxShape *pShape);
};


void SimpleGraspOfSoftCloth::initPhysics() {
    bool connected = m_robotSim.connect(m_guiHelper);
    b3Printf("robotSim connected = %d", connected);

    createWorld();
    createGround();
    createObstructingBox();
    createGripper();
    createCloth();
    m_guiHelper->autogenerateGraphicsObjects(m_dynamicsWorld);
}

void SimpleGraspOfSoftCloth::createWorld() {
    m_guiHelper->setUpAxis(2);

///    setupCollisionDetection
    /* 3D collision detection can become very complex, especially when using specialized algorithms.
     * That is why we work in two phases:
     *  1. Find collision of objects which are close to each other -> broad phase
     *  2. Use a more accurate collision algorithm for those objects close to each other -> collision dispatcher
     *  */

    /* The collision configuration specifies how memory will be allocated and managed for collision detection */
    m_collisionConfiguration = new btSoftBodyRigidBodyCollisionConfiguration();
    /* The broadphase checks which objects collides and which don't for a quick raw view of colliding objects */
    m_broadphase = new btDbvtBroadphase();
    /*  the dispatcher calculates collision of objects close to each other accurately */
    m_dispatcher = new btCollisionDispatcher(m_collisionConfiguration);

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
    /* btBoxShape *groundShape = createBoxShape(btVector3(btScalar(50.), btScalar(50.), btScalar(50.)));
     m_collisionShapes.push_back(groundShape);

     btTransform groundTransform;
     groundTransform.setIdentity();
     groundTransform.setOrigin(btVector3(0, -50, 0));

     {
         btScalar mass(0.);
         createRigidBody(mass, groundTransform, groundShape, btVector4(0, 0, 1, 1));
     }*/

    {
        b3RobotSimLoadFileArgs args("");
        args.m_fileName = "plane.urdf";
        args.m_startPosition.setValue(0, 0, 0);
        args.m_forceOverrideFixedBase = true;
        b3RobotSimLoadFileResults results;
        m_robotSim.loadFile(args, results);
        m_robotSim.setGravity(b3MakeVector3(0, 0, -10));
    }
}

void SimpleGraspOfSoftCloth::createObstructingBox() {
    btBoxShape *colShape = new btBoxShape(btVector3(.1, .1, .1));
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
        args.m_startOrientation.setEulerZYX(SIMD_PI, 0, 0);
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

/**
 * Put joints between the fingers and the motor so the motor will move the fingers. Weird that this isnt by default there.
 */
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

    int bodyIndex = 1;
    int motor_left_hinge_joint = 2;
    int motor_right_hinge_joint = 3;
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
    m_guiHelper->syncPhysicsToGraphics(m_dynamicsWorld);
    m_guiHelper->render(m_dynamicsWorld);

    m_robotSim.renderScene();
}

void SimpleGraspOfSoftCloth::exitPhysics() {
    m_robotSim.disconnect();
}

void SimpleGraspOfSoftCloth::stepSimulation(float deltaTime) {
    doMotorControl();
    m_robotSim.stepSimulation();
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

btRigidBody *SimpleGraspOfSoftCloth::createRigidBody(btScalar mass, btTransform transform, btBoxShape *pShape) {
    btAssert((!pShape || pShape->getShapeType() != INVALID_SHAPE_PROXYTYPE));

    //rigidbody is dynamic if and only if mass is non zero, otherwise static
    bool isDynamic = (mass != 0.f);

    btVector3 localInertia(0, 0, 0);
    if (isDynamic)
        pShape->calculateLocalInertia(mass, localInertia);

    //using motionstate is recommended, it provides interpolation capabilities, and only synchronizes 'active' objects

#define USE_MOTIONSTATE 1
#ifdef USE_MOTIONSTATE
    btDefaultMotionState *myMotionState = new btDefaultMotionState(transform);

    btRigidBody::btRigidBodyConstructionInfo cInfo(mass, myMotionState, pShape, localInertia);

    btRigidBody *body = new btRigidBody(cInfo);
    //body->setContactProcessingThreshold(m_defaultContactProcessingThreshold);

#else
    btRigidBody* body = new btRigidBody(mass, 0, shape, localInertia);
        body->setWorldTransform(startTransform);
#endif//

    body->setUserIndex(-1);
    m_dynamicsWorld->addRigidBody(body);
    return body;
}

class CommonExampleInterface *ET_SimpleGraspOfSoftClothCreateFunc(CommonExampleOptions &options) {
    return new SimpleGraspOfSoftCloth(options.m_guiHelper, options.m_option);
}
