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

class SimpleGraspOfSoftCloth : public CommonExampleInterface {
    CommonGraphicsApp *m_app;
    GUIHelperInterface *m_guiHelper;

    b3RobotSimAPI m_robotSim;
    int m_options;
    int m_gripperIndex;

public:

    SimpleGraspOfSoftCloth(struct GUIHelperInterface *helper, int options = 0) {
        m_gripperIndex = -1;
        m_app = helper->getAppInterface();
        m_guiHelper = helper;
        m_options = options;
    }

    virtual ~SimpleGraspOfSoftCloth() {
        m_app->m_renderer->enableBlend(false);
    }

    virtual void physicsDebugDraw(int debugDrawMode) {
        m_robotSim.debugDraw(debugDrawMode);
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

    virtual void resetCamera() {
        float dist = 1.24;
        float pitch = 47.4;
        float yaw = 25.2;
        float targetPos[3] = {-0.5, 0.40, -0.45};
        if (m_app->m_renderer && m_app->m_renderer->getActiveCamera()) {
            m_app->m_renderer->getActiveCamera()->setCameraDistance(dist);
            m_app->m_renderer->getActiveCamera()->setCameraPitch(pitch);
            m_app->m_renderer->getActiveCamera()->setCameraYaw(yaw);
            m_app->m_renderer->getActiveCamera()->setCameraTargetPosition(targetPos[0], targetPos[1], targetPos[2]);
        }
    }


    virtual void initPhysics() {
        bool connected = m_robotSim.connect(m_guiHelper);
        b3Printf("robotSim connected = %d", connected);

        createWorld();
        createGround();
//        createObstructingBox();
//        dropBoxFromHeight(1);
        createGripper();
        createCloth();
    }

    void createGround() {
        b3Printf("Creating ground plane...");
        {
            b3RobotSimLoadFileArgs args("");
            args.m_fileName = "plane.urdf";
            args.m_startPosition.setValue(0, 0, -0.1);
            args.m_startOrientation.setEulerZYX(0, 0, 0);
            args.m_forceOverrideFixedBase = true;
            args.m_useMultiBody = true;
            b3RobotSimLoadFileResults results;
            m_robotSim.loadFile(args, results);

        }
    }

    void createObstructingBox() {
        b3Printf("Creating an obstructing box...");
        b3Vector3 boxLocation = b3MakeVector3(0, 0, 0);
        b3Vector3 boxSize = b3MakeVector3(0.2, 0.1, 0.2);
        double mass = 100.0;
        m_robotSim.loadPrimitiveRigidBody(boxLocation, boxSize, mass);
    }


    void dropBoxFromHeight(int height) {
        b3Vector3 boxLocation = b3MakeVector3(0, 0, height);
        b3Vector3 boxSize = b3MakeVector3(0.05, 0.05, 0.05);
        double mass = 500.5;
        m_robotSim.loadPrimitiveRigidBody(boxLocation, boxSize, mass);
    }

    void createGripper() {
        b3Printf("Creating gripper...");
        addSliderControls();
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
    void addJointBetweenFingerAndMotor() {
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

        int bodyIndex = m_gripperIndex;
        int motor_left_hinge_joint = 2;
        int motor_right_hinge_joint = 3;
        int gripper_left_hinge_joint = 4;
        int gripper_right_hinge_joint = 6;
        m_robotSim.createJoint(bodyIndex, motor_left_hinge_joint, bodyIndex, gripper_left_hinge_joint, &revoluteJoint1);
        m_robotSim.createJoint(bodyIndex, motor_right_hinge_joint, bodyIndex, gripper_right_hinge_joint,
                               &revoluteJoint2);
    }

    void addSliderControls() const {
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

    void createCloth() {
        b3Printf("Creating patch of cloth...");
        m_robotSim.createClothPatch();
//        m_robotSim.loadBunny(0.1, 1, 0.02);
    }

    virtual void renderScene() {
        m_robotSim.renderScene();
    }

    virtual void exitPhysics() {
        m_robotSim.disconnect();
    }

    virtual void stepSimulation(float deltaTime) {
        doMotorControl();
        m_robotSim.stepSimulation();
    }

    void doMotorControl() {
        if (m_gripperIndex != -1) {
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
    }

    void createWorld() {
        int z_axis_is_up = 2;
        m_app->setUpAxis(z_axis_is_up);
        m_robotSim.setGravity(b3MakeVector3(0, 0, -2.5));
    }
};

/*
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
*/

class CommonExampleInterface *ET_SimpleGraspOfSoftClothCreateFunc(CommonExampleOptions &options) {
    return new SimpleGraspOfSoftCloth(options.m_guiHelper, options.m_option);
}
