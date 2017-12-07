#include "TeddyFromPybullet.h"

#include "btBulletDynamicsCommon.h"
#include "LinearMath/btVector3.h"
#include "LinearMath/btAlignedObjectArray.h"
#include "../CommonInterfaces/CommonRigidBodyBase.h"

#include "../../Utils/b3ResourcePath.h"
#include "Bullet3Common/b3FileUtils.h"
#include "../Importers/ImportObjDemo/LoadMeshFromObj.h"
#include "../OpenGLWindow/GLInstanceGraphicsShape.h"

#include "BulletSoftBody/btSoftRigidDynamicsWorld.h"
#include "BulletSoftBody/btSoftBodyHelpers.h"
#include "BulletSoftBody/btSoftBodyRigidBodyCollisionConfiguration.h"

#include "../SharedMemory/PhysicsClientC_API.h"
#include "../SharedMemory/PhysicsDirectC_API.h"
#include "../SharedMemory/SharedMemoryInProcessPhysicsC_API.h"


struct TeddyFromPybulletExample : public CommonRigidBodyBase {
   TeddyFromPybulletExample(struct GUIHelperInterface* helper):CommonRigidBodyBase(helper) {}
   virtual ~TeddyFromPybulletExample(){}
   virtual void initPhysics();
   virtual void renderScene();
   virtual void createEmptyDynamicsWorld();
   //virtual btSoftRigidDynamicsWorld* getSoftDynamicsWorld();

   virtual btSoftRigidDynamicsWorld*   getSoftDynamicsWorld()
      {
          //just make it a btSoftRigidDynamicsWorld please
          //or we will add type checking
          return (btSoftRigidDynamicsWorld*) m_dynamicsWorld;
      }

    void resetCamera() {
        float dist = 10;
        float pitch = 52;
        float yaw = 35;
        float targetPos[3]={0,0.46,0};
        m_guiHelper->resetCamera(dist,pitch,yaw,targetPos[0],targetPos[1],targetPos[2]);
    }
    btSoftBodyWorldInfo softBodyWorldInfo;
};

void TeddyFromPybulletExample::initPhysics() {
   m_guiHelper->setUpAxis(1);
   createEmptyDynamicsWorld();
   m_guiHelper->createPhysicsDebugDrawer(m_dynamicsWorld);

   if (m_dynamicsWorld->getDebugDrawer())
        m_dynamicsWorld->getDebugDrawer()->setDebugMode(btIDebugDraw::DBG_DrawWireframe+btIDebugDraw::DBG_DrawContactPoints);

    btBoxShape* groundShape = createBoxShape(btVector3(btScalar(50.),btScalar(50.),btScalar(50.)));
    m_collisionShapes.push_back(groundShape);

    btTransform groundTransform;
    groundTransform.setIdentity();
    groundTransform.setOrigin(btVector3(0,-50,0));

    {
        btScalar mass(0.);
        createRigidBody(mass,groundTransform,groundShape, btVector4(0,0,1,1));
    }

    // Load Soft Body by going through the PhysicsServerCommandProcessor
    b3PhysicsClientHandle sm = 0;
    int bodyUniqueId = -1;
    b3SharedMemoryStatusHandle statusHandle;
    int statusType;
    b3SharedMemoryCommandHandle command = b3CreateSoftBodyCommandInit(sm);
    b3SubmitClientCommandAndWaitStatus(sm , command);
    statusType = b3GetStatusType(statusHandle);
    bodyUniqueId = b3GetStatusBodyIndex(statusHandle);

    m_guiHelper->autogenerateGraphicsObjects(m_dynamicsWorld);
}

void TeddyFromPybulletExample::createEmptyDynamicsWorld()
    {
        m_collisionConfiguration = new btSoftBodyRigidBodyCollisionConfiguration();
        m_dispatcher = new  btCollisionDispatcher(m_collisionConfiguration);

        m_broadphase = new btDbvtBroadphase();

        m_solver = new btSequentialImpulseConstraintSolver;

        m_dynamicsWorld = new btSoftRigidDynamicsWorld(m_dispatcher, m_broadphase, m_solver, m_collisionConfiguration);
        m_dynamicsWorld->setGravity(btVector3(0, -10, 0));

        softBodyWorldInfo.m_broadphase = m_broadphase;
        softBodyWorldInfo.m_dispatcher = m_dispatcher;
        softBodyWorldInfo.m_gravity = m_dynamicsWorld->getGravity();
        softBodyWorldInfo.m_sparsesdf.Initialize();
    }

void TeddyFromPybulletExample::renderScene()
{
    CommonRigidBodyBase::renderScene();
    btSoftRigidDynamicsWorld* softWorld = getSoftDynamicsWorld();

    for (  int i=0;i<softWorld->getSoftBodyArray().size();i++)
    {
            btSoftBody* psb=(btSoftBody*)softWorld->getSoftBodyArray()[i];
            //if(softWorld->getDebugDrawer() && !(softWorld->getDebugDrawer()->getDebugMode() & (btIDebugDraw::DBG_DrawWireframe)))
            {
                btSoftBodyHelpers::DrawFrame(psb,softWorld->getDebugDrawer());
                btSoftBodyHelpers::Draw(psb,softWorld->getDebugDrawer(),softWorld->getDrawFlags());
            }
    }
}

CommonExampleInterface*    ET_TeddyFromPybulletCreateFunc(CommonExampleOptions& options)
{
   return new TeddyFromPybulletExample(options.m_guiHelper);
}
