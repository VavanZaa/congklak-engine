"""Saved historical evidence only; no native/provider/model construction."""
import copy,json,unittest
from unittest.mock import patch
from congklak import ownership_resolution_provenance as evidence


class ProvenanceTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tasks=evidence.plan()
  cls.units=[evidence.unit(t) for t in cls.tasks]
  cls.inventory=evidence.inventory()
  cls.overlay=evidence.final_projection(cls.units)

 def refuse(self,fn,*args):
  with self.assertRaises((ValueError,AssertionError,RuntimeError,KeyError,IndexError,TypeError)):
   fn(*args)

 def test_finite_actual_queries_and_broad_unknown(self):
  i=self.inventory
  self.assertEqual(i['actual_current_kind_counts'],dict(calibration=56,handoff=396,timing=240))
  self.assertEqual(i['actual_current_query_episodes'],692)
  self.assertEqual(i['actual_current_run_namespaces'],16)
  self.assertTrue(i['durable_actual_intents_paired_one_to_one'])
  self.assertEqual(sum(x['durable_intent_count'] for x in i['actual_run_query_coverage']),692)
  self.assertTrue(all(x['one_to_one'] and x['durable_intent_count']==x['actual_archive_count']
   for x in i['actual_run_query_coverage']))
  self.assertEqual(len({u['episode'] for u in self.units}),692)
  self.assertEqual(sum(t['coverage_proof'] is not None for t in self.tasks),272)
  broad=next(x for x in i['missing_attempts'] if x['id']=='broad-historical-native-internals')
  self.assertTrue(broad['permanently_unknown'])
  self.assertIn('not comprehensively recorded',broad['limitation'])
  self.assertIsNone(broad['exact_missing_attempt_count'])
  self.assertFalse(i['historical_provenance_complete'])
  self.assertFalse(i['historically_unseen_claim'])
  self.assertTrue(i['unknown_queries_are_not_proof_of_gradient_use'])
  smokes=next(x for x in i['missing_attempts'] if x['id']=='ad-hoc-native-smokes')
  self.assertEqual(smokes['accounting_discrepancy']['reported_smoke_group'],11)
  self.assertEqual(smokes['accounting_discrepancy']['residual_smokes_from_breakdown'],10)
  self.assertFalse(smokes['accounting_discrepancy']['reconciled'])
  self.assertTrue(all(x['actual_query'] is False for x in i['excluded_nonquery_artifacts']))

 def test_zero_query_origins_and_all_high_domain(self):
  o=self.overlay
  self.assertEqual(o['actual_declared_origins'],27)
  self.assertEqual(o['zero_query_schedule_origins'],11)
  for x in o['declared_schedule_origins']:
   self.assertIn(x['family'],o['family_components'])
   if not x['actual_query_episodes']:
    self.assertEqual(o['origin_quotients'][x['family']],[])
  qs=[bytes.fromhex(q) for q in o['quotient_components']]
  self.assertTrue(all(sum(q[:14])>=25 for q in qs))
  self.assertTrue(any(sum(q[:14])>=75 for q in qs))
  timing={u['family'] for u in self.units if u['decision_id'].startswith('timing/')}
  self.assertEqual(len(timing),1)
  self.assertEqual(len(o['actual_episode_units']),692)
  self.assertFalse(o['optimizer_use_inferred_from_engineering_exposure'])

 def test_resealed_unit_and_task_changes_refuse(self):
  t=self.tasks[0];u=self.units[0]
  for key,value in [('event_count',u['event_count']+1),('family',u['family']+'-wrong'),
                    ('declared_origin',u['declared_origin']+'-wrong')]:
   bad=copy.deepcopy(u);bad[key]=value;self.refuse(evidence.validate_unit,t,bad)
  bad=copy.deepcopy(u);bad['activity']['native_searches']=False
  self.refuse(evidence.validate_unit,t,bad)
  bad=copy.deepcopy(u);bad['owned_records'][0][1]+=1
  self.refuse(evidence.validate_unit,t,bad)
  for key,value in [('index',False),('declared_origin','fabricated'),('source_sha256','0'*64)]:
   bad=copy.deepcopy(t);bad[key]=value;self.refuse(evidence.unit,bad)
  bad=copy.deepcopy(t);bad['attempt_intent']['sha256']='0'*64
  self.refuse(evidence.unit,bad)

 def test_missing_intent_or_archive_refuses(self):
  original=dict(evidence._accepted())
  try:
   for missing in (self.tasks[0]['source'],self.tasks[0]['attempt_intent']['path']):
    changed=dict(original);del changed[missing];evidence._layout.cache_clear()
    with patch.object(evidence,'_accepted',lambda:changed):self.refuse(evidence.plan)
  finally:evidence._layout.cache_clear()

 def test_inventory_bool_int_and_missing_scope_refuse(self):
  for key,value in [('historical_provenance_complete',0),('actual_current_missing_query_archives',False)]:
   bad=copy.deepcopy(self.inventory);bad[key]=value;self.refuse(evidence.validate_inventory,bad)
  bad=copy.deepcopy(self.inventory);bad['missing_attempts']=bad['missing_attempts'][1:]
  self.refuse(evidence.validate_inventory,bad)

 def test_projection_rejects_resealed_invalid_unit(self):
  bad=copy.deepcopy(self.units);bad[0]['owned_records']=bad[0]['owned_records'][1:]
  self.refuse(evidence.validate_projection,self.overlay,bad)
  bad=copy.deepcopy(self.overlay);bad['historical_provenance_complete']=0
  self.refuse(evidence.validate_projection,bad,self.units)

 def test_validators_do_not_call_producer_entrypoints(self):
  def trap(*args,**kwargs):raise RuntimeError('producer was called')
  with patch.object(evidence,'unit',trap),patch.object(evidence,'inventory',trap),patch.object(evidence,'final_projection',trap):
   self.assertTrue(evidence.validate_inventory(self.inventory))
   for index in (0,396,420,452,691):
    self.assertTrue(evidence.validate_unit(self.tasks[index],self.units[index]))
   self.assertTrue(evidence.validate_projection(self.overlay,self.units))


if __name__=='__main__':unittest.main()
