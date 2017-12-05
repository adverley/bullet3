#include "SoftBodyFromObj.h"

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
 
#include "Wavefront/tiny_obj_loader.h"


struct SoftBodyFromObjExample : public CommonRigidBodyBase {
   SoftBodyFromObjExample(struct GUIHelperInterface* helper):CommonRigidBodyBase(helper) {}
   virtual ~SoftBodyFromObjExample(){}
   virtual void initPhysics();
   virtual void renderScene();
   virtual void createEmptyDynamicsWorld();
   //virtual btSoftRigidDynamicsWorld* getSoftDynamicsWorld();
   
   /*
   void createEmptyDynamicsWorld()
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
    */

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

void SoftBodyFromObjExample::initPhysics() {
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
    

    // load our obj mesh
    const char* fileName = "teddy.obj";//sphere8.obj";//sponza_closed.obj";//sphere8.obj";
    char relativeFileName[1024];
    if (b3ResourcePath::findResourcePath(fileName, relativeFileName, 1024)) {
        char pathPrefix[1024];
        b3FileUtils::extractPath(relativeFileName, pathPrefix, 1024); 
    }
    std::vector<tinyobj::shape_t> shapes;
    std::string err = tinyobj::LoadObj(shapes, relativeFileName, "");

    btAlignedObjectArray<btScalar> vertices;
    btAlignedObjectArray<int> indices;

    //loop through all the shapes and add vertices and indices
    int offset = 0;
    for(int i=0;i<shapes.size();++i) {
        const tinyobj::shape_t& shape = shapes[i];
    
        //add vertices
        for(int j=0;j<shape.mesh.positions.size();++j) {
            vertices.push_back(shape.mesh.positions[j]);
        }

        //add indices
        for(int j=0;j<shape.mesh.indices.size();++j) {
            indices.push_back(offset + shape.mesh.indices[j]);   
        }
        offset += shape.mesh.positions.size();
    }
    printf("[INFO] Obj loaded: Extracted %d vertices, %d indices from obj file [%s]\n", vertices.size(), indices.size(), fileName);

    btSoftBody* psb = btSoftBodyHelpers::CreateFromTriMesh(softBodyWorldInfo, &vertices[0], &(indices[0]), indices.size()/3);
   
    btVector3 scaling(0.1, 0.1, 0.1); 
       
    btSoftBody::Material* pm=psb->appendMaterial();
    pm->m_kLST =   0.75;
    pm->m_flags -= btSoftBody::fMaterial::DebugDraw;
    psb->scale(scaling);
    psb->generateBendingConstraints(4,pm);
    psb->m_cfg.piterations = 2;  
    psb->m_cfg.kDF = 0.75;
    psb->m_cfg.collisions |= btSoftBody::fCollision::VF_SS;  
    psb->randomizeConstraints();  

    btMatrix3x3    m;
    btVector3 x(0,10,0);
    btVector3 a(0,0,0);
    m.setEulerZYX(a.x(),a.y(),a.z());
    psb->transform(btTransform(m,x));  
    psb->setTotalMass(1); 
    psb->getCollisionShape()->setMargin(0.1f);
    getSoftDynamicsWorld()->addSoftBody(psb);
    m_guiHelper->autogenerateGraphicsObjects(m_dynamicsWorld);
}

void SoftBodyFromObjExample::createEmptyDynamicsWorld()
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

/*
virtual btSoftRigidDynamicsWorld*   getSoftDynamicsWorld()
    {
        //just make it a btSoftRigidDynamicsWorld please
        //or we will add type checking
        return (btSoftRigidDynamicsWorld*) m_dynamicsWorld;
    }
*/

void SoftBodyFromObjExample::renderScene()
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

CommonExampleInterface*    ET_SoftBodyFromObjCreateFunc(CommonExampleOptions& options)
{
   return new SoftBodyFromObjExample(options.m_guiHelper);
}
