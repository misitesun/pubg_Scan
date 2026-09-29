import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import json
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace
import cv2
import numpy as np
import onnx
from onnx import helper,numpy_helper,TensorProto
from vision.dataset import Project,save_image,source_digest
from vision.detector import Detection,Detector
from vision.identity import IdentityFilter,TeamMarkers,ROLES


class IdentityTests(unittest.TestCase):
    def test_confirmed_dead_person_excluded_without_losing_self_or_other_targets(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);p=Project(root/'project')
            for i in range(2):
                image=root/f'{i}.jpg';save_image(image,np.full((80,100,3),i*80,np.uint8));p.import_images([image])
                p.annotate(i,[[10,10,40,60],[50,15,80,50]],['self','opponent'])
                p.samples[i]['user_confirmed']=True
            p.samples[0]['ignored_box_indices']=[1];p.save()
            # 修改身份后，死亡排除仍对应原身体；删除其他框后索引随身体重新关联。
            p.annotate(0,p.samples[0]['boxes'],['self','opponent'])
            self.assertEqual(p.samples[0]['ignored_box_indices'],[1])
            yaml,_,_=p.export_dataset(root/'data',identities=True,user_confirmed_only=True)
            labels=[int(line.split()[0]) for f in (yaml.parent/'labels').rglob('*.txt') for line in f.read_text().splitlines()]
            self.assertCountEqual(labels,[0,0,1])
            p.annotate(0,[p.samples[0]['boxes'][1]],['opponent'])
            self.assertEqual(p.samples[0]['ignored_box_indices'],[0])
            with self.assertRaises(ValueError):p.export_dataset(root/'invalid',identities=True,user_confirmed_only=True)

    def test_review_import_requires_original_image_and_boxes_and_explicit_confirmation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);p=Project(root/'project')
            for i in range(2):
                image=root/f'{i}.jpg';save_image(image,np.full((80,100,3),i*80,np.uint8));p.import_images([image])
                p.annotate(i,[[10,10,40,60]],['unknown'])
            entries=[dict(image=s['image'],image_sha256=source_digest(p.image_path(s)),boxes=s['boxes'],
                          roles=['opponent'],status='confirmed',note='用户修改') for s in p.samples]
            record=dict(format='person-vision-review-v1',samples=entries)
            path=root/'review.json'
            entries[1]['image_sha256']='stale';path.write_text(json.dumps(record))
            before=p.manifest.read_bytes()
            with self.assertRaises(ValueError):p.import_identity_review(path)
            self.assertEqual(before,p.manifest.read_bytes())
            self.assertEqual(p.samples[0]['roles'],['unknown'])
            entries[1]['image_sha256']=source_digest(p.image_path(p.samples[1]))
            entries[1]['status']='pending';path.write_text(json.dumps(record))
            self.assertEqual(p.import_identity_review(path),1)
            self.assertTrue(p.samples[0]['user_confirmed'])
            self.assertFalse(p.samples[1]['roles_reviewed'])
            with self.assertRaises(ValueError):p.export_dataset(root/'formal',identities=True,user_confirmed_only=True)
            p.samples[1]['reviewed']=False;p.save()
            entries[1]['status']='confirmed';path.write_text(json.dumps(record))
            self.assertEqual(p.import_identity_review(path),2)
            self.assertTrue(p.samples[1]['reviewed'])
            _,nt,nv=p.export_dataset(root/'formal',identities=True,user_confirmed_only=True)
            self.assertEqual(nt+nv,2)
            entries[0]['boxes']=[[0,0,40,60]];path.write_text(json.dumps(record))
            with self.assertRaises(ValueError):p.import_identity_review(path)

    def test_unknown_and_person_never_become_opponent(self):
        policy=IdentityFilter()
        frame=np.zeros((160,200,3),np.uint8)
        box=Detection(40,40,80,120,.99)
        for i in range(8):self.assertEqual(policy.update(frame,[box],i*.1),[])
        self.assertEqual(policy.decisions[0]['role'],'unknown')

    def test_stability_conflict_and_occlusion_expiry(self):
        p=IdentityFilter();f=np.zeros((160,200,3),np.uint8)
        b=Detection(40,40,80,120,.8,'opponent',1)
        self.assertEqual(p.update(f,[b],0),[])
        self.assertEqual(p.update(f,[b],.1),[])
        self.assertEqual(len(p.update(f,[b],.2)),1)
        self.assertEqual(p.update(f,[replace(b,role='self')],.3),[])
        for t in (.4,.5,.6):self.assertEqual(p.update(f,[b],t),[])
        # 身份被否定后，即使模型又预测对手也不能在原轨迹上画框。
        self.assertEqual(p.update(f,[b],1.2),[])
        self.assertEqual(p.update(f,[b],1.3),[])
        self.assertEqual(len(p.update(f,[b],1.4)),1)
        # 切镜后不能把连续帧计数带到新场景。
        self.assertEqual(p.update(np.full_like(f,255),[b],1.5),[])

    def test_team_marker_near_person_and_ambiguous_association(self):
        rng=np.random.default_rng(42)
        template=rng.integers(0,255,(12,12,3),dtype=np.uint8)
        markers=TeamMarkers();markers.templates=[template]
        f=np.zeros((200,200,3),np.uint8);f[55:67,84:96]=template
        a=Detection(75,70,110,180,.99,'opponent',1)
        b=Detection(130,70,165,180,.99,'opponent',1)
        confirmed,ambiguous=markers.associate(f,[a,b])
        self.assertEqual(confirmed,{0});self.assertFalse(ambiguous)
        p=IdentityFilter(markers)
        for i in range(4):
            visible=p.update(f,[a,b],i*.1)
        self.assertEqual([x.x1 for x in visible],[130])
        self.assertEqual(p.decisions[0]['role'],'teammate')
        confirmed,ambiguous=markers.associate(f,[a,replace(a,x1=78,x2=113)])
        self.assertFalse(confirmed);self.assertEqual(ambiguous,{0,1})

    def test_identity_export_does_not_turn_self_unknown_into_background(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);project=Project(root/'project')
            for i in range(3):
                path=root/f'{i}.jpg';save_image(path,np.full((160,200,3),i*40,np.uint8))
                project.import_images([path])
            project.annotate(0,[[10,10,40,100],[60,20,90,100]],['self','opponent'])
            project.annotate(1,[[10,10,40,100],[60,20,90,100]],['unknown','opponent'])
            project.annotate(2,[[10,10,40,100]]) # 仅人物已复核，身份未复核
            yaml,nt,nv=project.export_dataset(root/'data',identities=True)
            self.assertEqual(nt+nv,2)
            classes=[int(line.split()[0]) for path in (yaml.parent/'labels').rglob('*.txt') for line in path.read_text().splitlines()]
            self.assertCountEqual(classes,[0,1,3,1])
            loaded=Project(project.root)
            self.assertFalse(loaded.samples[2]['roles_reviewed'])
            with self.assertRaises(ValueError):project.annotate(0,[[0,0,10,10]],['enemy'])
            with self.assertRaises(ValueError):project.annotate(0,[[0,0,np.nan,10]],['self'])

    def test_role_metadata_and_all_four_class_decode(self):
        with tempfile.TemporaryDirectory() as temp:
            values=np.zeros((1,8,100),np.float32)
            for i in range(4):
                values[0,:4,i]=[10+i*14,32,10,20]
                values[0,4+i,i]=.9
            graph=helper.make_graph([helper.make_node('Constant',[],['output'],value=numpy_helper.from_array(values))],
                'roles',[helper.make_tensor_value_info('images',TensorProto.FLOAT,[1,3,64,64])],
                [helper.make_tensor_value_info('output',TensorProto.FLOAT,[1,8,100])])
            model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',17)],ir_version=9)
            helper.set_model_props(model,{'task':'detect','names':repr(dict(enumerate(ROLES)))})
            path=Path(temp)/'roles.onnx';onnx.save(model,path)
            detector=Detector(path)
            self.assertTrue(detector.identity_model)
            boxes,_=detector.detect(np.zeros((64,64,3),np.uint8))
            self.assertCountEqual([b.role for b in boxes],list(ROLES))

    def test_teammate_model_guess_is_not_confirmed_team_marker(self):
        policy=IdentityFilter();frame=np.zeros((160,200,3),np.uint8)
        box=Detection(40,40,80,120,.99,'teammate',2)
        self.assertFalse(policy.update(frame,[box],0))
        self.assertEqual(policy.decisions[0]['role'],'unknown')

if __name__=='__main__':unittest.main()
