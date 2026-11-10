Require Import String.
Require Import Lia.
Require Import stdpp.gmap.

Module StdppCompat.
  Notation lookup_insert_eq := lookup_insert.
End StdppCompat.
Import StdppCompat.

Definition var := string.
Definition loc := nat.
Definition timestamp := nat.
Definition val := Z.

Inductive instr :=
| Load (t1 : timestamp) (x : var) (l : loc) (t : timestamp)    (* at t1: x <- (l, t)  where  t ≤ t1 *)
(* t must be less than t1, unless l belongs to the issuing node *)
| Store (t1 : timestamp) (l : loc) (t : timestamp) (v : val)   (* at t1: (l, t) <- v  where t1 ≤ t *)
(* likewise, they can only be equal if l belongs to the issuing node *)
.
(* expectation: each node issues instructions with monotonic t1's *)
(* store is non-blocking, load is blocking -- waits for it to become readable because nobody is writing
   it anymore *)
(* for every good program, there should exist a valid schedule *)
(* to signal that there will be no more stores to a location/timestamp:

  Store ... l t1
  Store t2 (t2 > t1) then this node will no longer issue stores that come from t1

  barrier indicates that the node will not send any more instructions
*)
(* when the runtime knows that all nodes will not write (l, t), then (l, t) becomes readable *)

Section semantics.

Definition node_id := nat.

Record comm_graph := {
  world_size : nat;

  (* The runtime retains two directed logical-time bounds for each reachable
     channel.  readable_lag is the minimum distance: the smallest permitted
     future write determines when no more writes can arrive and a version may
     become Readable.  discard_lag is the maximum distance: the largest
     permitted look-back determines when no more reads can arrive and a version
     may eventually become Discardable.  Ordinary graphs use the same value for
     both; combining graphs can make them differ.  These are operation-time
     bounds, not physical message-delivery latency. *)
  readable_lag : node_id -> node_id -> option nat;
  discard_lag : node_id -> node_id -> option nat;

  channel_in_bounds :
    forall p q d,
      readable_lag p q = Some d ->
      p < world_size /\ q < world_size;

  channel_irreflexive : 
    forall p, p < world_size -> 
      readable_lag p p = None;

  channel_positive : 
    forall p q d, readable_lag p q = Some d -> d > 0;

  discard_channel :
    forall p q,
      (exists d, readable_lag p q = Some d) <->
      (exists d, discard_lag p q = Some d);

  channel_discard_bounds :
    forall p q readable discardable,
      readable_lag p q = Some readable ->
      discard_lag p q = Some discardable ->
      readable <= discardable
}.

Definition symmetric_graph (G : comm_graph) : Prop := 
  forall p q,
    p < world_size G -> 
    q < world_size G ->
    readable_lag G p q = readable_lag G q p /\
    discard_lag G p q = discard_lag G q p.

Definition neighbors (G : comm_graph) (p q : node_id) : Prop :=
  exists d, readable_lag G p q = Some d.

Definition target_allowed (G : comm_graph) (issuer target : node_id) : Prop :=
  issuer = target \/ neighbors G issuer target.

(* local operations may access the issuing
   timestamp, while remote loads must target the past and remote stores
   must target the future. *)
Definition operation_allowed
    (owner : loc -> node_id) (issuer : node_id) (op : instr) (G : comm_graph) : Prop :=
  match op with
  | Load issued_at _ l target_at =>
      target_allowed G issuer (owner l) /\
      if Nat.eqb (owner l) issuer then
        target_at <= issued_at
      else 
        exists readable discardable,
          readable_lag G issuer (owner l) = Some readable /\
          discard_lag G issuer (owner l) = Some discardable /\
          target_at + readable <= issued_at /\
          issued_at <= target_at + discardable
  | Store issued_at l target_at _ =>
      target_allowed G issuer (owner l) /\
      if Nat.eqb (owner l) issuer
      then 
        issued_at <= target_at
      else
        exists d,
          readable_lag G issuer (owner l) = Some d /\
          issued_at + d <= target_at
  end.

Variable owner : loc -> node_id.
Variable G : comm_graph.

Definition env := gmap var val.
Definition node_pool := gmap node_id (list instr * env).

(* A version is visible when no unexecuted store in any node's
   remaining program can still write to that location and timestamp.
   This is a specification-level condition; the runtime may use
   timestamps and heartbeats to determine it without inspecting programs. *)
Definition writes_to (l : loc) (t : timestamp) (op : instr) : Prop :=
  match op with
  | Load _ _ _ _ => False
  | Store _ l' t' _ => l' = l /\ t' = t
  end.

Definition pending_write
    (np : node_pool) (l : loc) (t : timestamp) : Prop :=
  exists i instructions e op,
    np !! i = Some (instructions, e) /\
    op ∈ instructions /\
    writes_to l t op.

(* A version becomes readable only after every program instruction 
   capable of writing that version has executed. *)
Definition visible
    (np : node_pool) (l : loc) (t : timestamp) : Prop :=
  ~ pending_write np l t.

Definition reads_from (l : loc) (t : timestamp) (op : instr) : Prop :=
  match op with
  | Load _ _ l' t' => l' = l /\ t' = t
  | Store _ _ _ _ => False
  end.

Definition pending_read
    (np : node_pool) (l : loc) (t : timestamp) : Prop :=
  exists i instructions e op,
    np !! i = Some (instructions, e) /\
    op ∈ instructions /\
    reads_from l t op.

Inductive mem_status := Writeable | Readable | Discardable.
Definition mem := gmap (loc * timestamp) (val * mem_status).
Definition config := (node_pool * mem)%type.

Inductive step : config -> config -> Prop :=
| step_load np i t1 x l t rest e m v (Hi: np !! i = Some (Load t1 x l t :: rest, e))
  (Hcheck : operation_allowed owner i (Load t1 x l t) G)
  (Hload: m !! (l, t) = Some (v, Readable)):
    step (np, m) (<[i := (rest, <[x := v]>e)]>np, m)
| step_store np i t1 l t v rest e m
  (Hi : np !! i = Some (Store t1 l t v :: rest, e))
  (Hcheck : operation_allowed owner i (Store t1 l t v) G)
  (Hslot : m !! (l, t) = None \/
    exists v0, m !! (l, t) = Some (v0, Writeable)):
    step (np, m) (<[i := (rest, e)]>np, <[(l, t) := (v, Writeable)]>m)
| step_readable np m l t v
  (Hload: m !! (l, t) = Some (v, Writeable))
  (Hvisible: visible np l t):
    step (np, m) (np, <[(l, t) := (v, Readable)]>m)
| step_discardable np m l t v
  (Hload : m !! (l, t) = Some (v, Readable))
  (Hretired : ~ pending_read np l t):
    step (np, m) (np, <[(l, t) := (v, Discardable)]>m)
(* It's the runtime's responsibility to ensure that this is done at the right time to make the program work. *)
.

Inductive evict : config -> config -> Prop :=
| evict_discardable np m l t v
    (Hdiscardable : m !! (l, t) = Some (v, Discardable))
    (Hunused : ~ pending_write np l t) :
    evict (np, m) (np, delete (l, t) m).

Inductive runtime_step : config -> config -> Prop :=
| runtime_step_model c c' (Hstep : step c c') : runtime_step c c'
| runtime_step_evict c c' (Hevict : evict c c') : runtime_step c c'.

Definition observationally_equivalent (c c' : config) : Prop :=
  fst c = fst c'.

Definition init_config (prog : gmap node_id (list instr)) : config := ((λ i, (i, ∅)) <$> prog, ∅).

(* I added some proofs for some random basic properties *)

(* Any load transition requires the accessed version to be Readable. *)
Lemma load_only_reads_readable
    (np : node_pool) (i : node_id)
    (t1 : timestamp) (x : var) (l : loc) (t : timestamp)
    (rest : list instr) (e : env) (m : mem) (v : val)
    (Hi : np !! i = Some (Load t1 x l t :: rest, e))
    (Hcheck : operation_allowed owner i (Load t1 x l t) G)
    (Hload : m !! (l, t) = Some (v, Readable)) :
  m !! (l, t) = Some (v, Readable).
Proof.
  exact Hload.
Qed.

(* A store target must be absent or already Writeable. *)
Lemma store_target_available
    (m : mem) (l : loc) (t : timestamp)
    (Hslot :
      m !! (l, t) = None \/
      exists v0, m !! (l, t) = Some (v0, Writeable)) :
  forall v, m !! (l, t) <> Some (v, Readable).
Proof.
  intros v Hread.
  destruct Hslot as [Habsent | [v0 Hwriteable]].
  - rewrite Hread in Habsent.
    discriminate.
  - rewrite Hread in Hwriteable.
    discriminate.
Qed.

Definition pending_stores_available (np : node_pool) (m : mem) : Prop :=
  forall l t,
    pending_write np l t ->
    m !! (l, t) = None \/
    exists v, m !! (l, t) = Some (v, Writeable).

Definition program_produces
    (prog : gmap node_id (list instr)) (l : loc) (t : timestamp) : Prop :=
  exists i instructions issued_at v,
    prog !! i = Some instructions /\
    Store issued_at l t v ∈ instructions.

Definition load_versions_produced (prog : gmap node_id (list instr)) : Prop :=
  forall i instructions issued_at x l t,
    prog !! i = Some instructions ->
    Load issued_at x l t ∈ instructions ->
    program_produces prog l t.

Definition pending_loads_available (np : node_pool) (m : mem) : Prop :=
  forall l t,
    pending_read np l t ->
    pending_write np l t \/
    exists v,
      m !! (l, t) = Some (v, Writeable) \/
      m !! (l, t) = Some (v, Readable).

Lemma pending_write_after_head_removal
    (np : node_pool) (i : node_id)
    (head : instr) (rest : list instr)
    (e e' : env) (l : loc) (t : timestamp)
    (Hi : np !! i = Some (head :: rest, e)) :
  pending_write (<[i := (rest, e')]> np) l t ->
  pending_write np l t.
Proof.
  intros Hpending.
  destruct Hpending as
    [j [instructions [ej [op [Hj [Hop Hwrites]]]]]].
  destruct (decide (j = i)) as [Heq | Hne].
  - subst j.
    assert (<[i := (rest, e')]> np !! i = Some (rest, e')) as Hins
      by apply lookup_insert.
    pose proof (eq_trans (eq_sym Hj) Hins) as Heqpair.
    inversion Heqpair.
    exists i, (head :: rest), e, op.
    repeat split.
    + exact Hi.
    + apply elem_of_cons. right. rewrite <- H0. exact Hop.
    + exact Hwrites.
  - assert (np !! j = Some (instructions, ej)) as Hj_old.
    { rewrite <- Hj. symmetry. apply lookup_insert_ne. congruence. }
    exists j, instructions, ej, op.
    repeat split; assumption.
Qed.

Lemma pending_read_after_head_removal
    (np : node_pool) (i : node_id)
    (head : instr) (rest : list instr)
    (e e' : env) (l : loc) (t : timestamp)
    (Hi : np !! i = Some (head :: rest, e)) :
  pending_read (<[i := (rest, e')]> np) l t ->
  pending_read np l t.
Proof.
  intros Hpending.
  destruct Hpending as
    [j [instructions [ej [op [Hj [Hop Hreads]]]]]].
  destruct (decide (j = i)) as [Heq | Hne].
  - subst j.
    assert (<[i := (rest, e')]> np !! i = Some (rest, e')) as Hins
      by apply lookup_insert.
    pose proof (eq_trans (eq_sym Hj) Hins) as Heqpair.
    inversion Heqpair.
    exists i, (head :: rest), e, op.
    repeat split.
    + exact Hi.
    + apply elem_of_cons. right. rewrite <- H0. exact Hop.
    + exact Hreads.
  - assert (np !! j = Some (instructions, ej)) as Hj_old.
    { rewrite <- Hj. symmetry. apply lookup_insert_ne. congruence. }
    exists j, instructions, ej, op.
    repeat split; assumption.
Qed.

Lemma pending_write_after_load
    np i issued_at x l t rest e e' ql qt
    (Hi : np !! i = Some (Load issued_at x l t :: rest, e)) :
  pending_write np ql qt ->
  pending_write (<[i := (rest, e')]> np) ql qt.
Proof.
  intros Hpending.
  destruct Hpending as
    [j [instructions [ej [op [Hj [Hop Hwrites]]]]]].
  destruct (decide (j = i)) as [-> | Hne].
  - rewrite Hi in Hj.
    injection Hj as <- <-.
    rewrite elem_of_cons in Hop.
    destruct Hop as [-> | Hop].
    + exact (False_rect _ Hwrites).
    + exists i, rest, e', op.
      repeat split.
      * apply lookup_insert.
      * exact Hop.
      * exact Hwrites.
  - exists j, instructions, ej, op.
    repeat split.
    + transitivity (np !! j).
      * apply lookup_insert_ne. congruence.
      * exact Hj.
    + exact Hop.
    + exact Hwrites.
Qed.

Lemma pending_write_after_other_store
    np i issued_at l t v rest e ql qt
    (Hi : np !! i = Some (Store issued_at l t v :: rest, e))
    (Hdifferent : (ql, qt) <> (l, t)) :
  pending_write np ql qt ->
  pending_write (<[i := (rest, e)]> np) ql qt.
Proof.
  intros Hpending.
  destruct Hpending as
    [j [instructions [ej [op [Hj [Hop Hwrites]]]]]].
  destruct (decide (j = i)) as [-> | Hne].
  - rewrite Hi in Hj.
    injection Hj as <- <-.
    rewrite elem_of_cons in Hop.
    destruct Hop as [-> | Hop].
    + cbn in Hwrites. destruct Hwrites as [-> ->]. contradiction.
    + exists i, rest, e, op.
      repeat split.
      * apply lookup_insert.
      * exact Hop.
      * exact Hwrites.
  - exists j, instructions, ej, op.
    repeat split.
    + transitivity (np !! j).
      * apply lookup_insert_ne. congruence.
      * exact Hj.
    + exact Hop.
    + exact Hwrites.
Qed.

Lemma pending_stores_available_step
    (np np' : node_pool) (m m' : mem) :
  pending_stores_available np m ->
  step (np, m) (np', m') ->
  pending_stores_available np' m'.
Proof.
  intros Hsafe Hstep.
  inversion Hstep; subst.
  - intros ql qt Hpending.
    apply Hsafe.
    eapply pending_write_after_head_removal; eauto.
  - intros ql qt Hpending.
    destruct (decide ((ql, qt) = (l, t))) as [Heq | Hne].
    + inversion Heq; subst.
      right. exists v. apply lookup_insert.
    + destruct (Hsafe ql qt ltac:(
        eapply pending_write_after_head_removal; eauto))
        as [Habsent | [v0 Hwriteable]].
      * left. transitivity (m !! (ql, qt)).
        -- apply lookup_insert_ne. congruence.
        -- exact Habsent.
      * right. exists v0. transitivity (m !! (ql, qt)).
        -- apply lookup_insert_ne. congruence.
        -- exact Hwriteable.
  - intros ql qt Hpending.
    destruct (decide ((ql, qt) = (l, t))) as [Heq | Hne].
    + inversion Heq; subst.
      exfalso. exact (Hvisible Hpending).
    + destruct (Hsafe ql qt Hpending) as [Habsent | [v0 Hwriteable]].
      * left. transitivity (m !! (ql, qt)).
        -- apply lookup_insert_ne. congruence.
        -- exact Habsent.
      * right. exists v0. transitivity (m !! (ql, qt)).
        -- apply lookup_insert_ne. congruence.
        -- exact Hwriteable.
  - intros ql qt Hpending.
    destruct (decide ((ql, qt) = (l, t))) as [Heq | Hne].
    + inversion Heq; subst.
      exfalso.
      destruct (Hsafe l t Hpending) as [Habsent | [v0 Hwriteable]].
      * rewrite Hload in Habsent. discriminate.
      * rewrite Hload in Hwriteable. discriminate.
    + destruct (Hsafe ql qt Hpending) as [Habsent | [v0 Hwriteable]].
      * left. transitivity (m !! (ql, qt)).
        -- apply lookup_insert_ne. congruence.
        -- exact Habsent.
      * right. exists v0. transitivity (m !! (ql, qt)).
        -- apply lookup_insert_ne. congruence.
        -- exact Hwriteable.
Qed.

Lemma pending_loads_available_step
    (np np' : node_pool) (m m' : mem) :
  pending_loads_available np m ->
  step (np, m) (np', m') ->
  pending_loads_available np' m'.
Proof.
  intros Havailable Hstep.
  inversion Hstep; subst.
  - intros ql qt Hread.
    destruct (Havailable ql qt ltac:(
      eapply pending_read_after_head_removal; eauto))
      as [Hwrite | [v0 [Hwriteable | Hreadable]]].
    + left. eapply pending_write_after_load; eauto.
    + right. exists v0. left. exact Hwriteable.
    + right. exists v0. right. exact Hreadable.
  - intros ql qt Hread.
    destruct (Havailable ql qt ltac:(
      eapply pending_read_after_head_removal; eauto))
      as [Hwrite | [v0 [Hwriteable | Hreadable]]].
    + destruct (decide ((ql, qt) = (l, t))) as [Heq | Hne].
      * right. exists v. left. inversion Heq; subst. apply lookup_insert.
      * left. eapply pending_write_after_other_store; eauto.
    + right.
      destruct (decide ((ql, qt) = (l, t))) as [Heq | Hne].
      * exists v. left. inversion Heq; subst. apply lookup_insert.
      * exists v0. left.
        transitivity (m !! (ql, qt)).
        -- apply lookup_insert_ne. congruence.
        -- exact Hwriteable.
    + right.
      destruct (decide ((ql, qt) = (l, t))) as [Heq | Hne].
      * exists v. left. inversion Heq; subst. apply lookup_insert.
      * exists v0. right.
        transitivity (m !! (ql, qt)).
        -- apply lookup_insert_ne. congruence.
        -- exact Hreadable.
  - intros ql qt Hread.
    destruct (Havailable ql qt Hread)
      as [Hwrite | [v0 [Hwriteable | Hreadable]]].
    + destruct (decide ((ql, qt) = (l, t))) as [Heq | Hne].
      * inversion Heq; subst. exfalso. exact (Hvisible Hwrite).
      * left. exact Hwrite.
    + right.
      destruct (decide ((ql, qt) = (l, t))) as [Heq | Hne].
      * inversion Heq; subst.
        rewrite Hload in Hwriteable.
        injection Hwriteable as <-.
        exists v. right. apply lookup_insert.
      * exists v0. left.
        transitivity (m !! (ql, qt)).
        -- apply lookup_insert_ne. congruence.
        -- exact Hwriteable.
    + right.
      destruct (decide ((ql, qt) = (l, t))) as [Heq | Hne].
      * inversion Heq; subst. rewrite Hload in Hreadable. discriminate.
      * exists v0. right.
        transitivity (m !! (ql, qt)).
        -- apply lookup_insert_ne. congruence.
        -- exact Hreadable.
  - intros ql qt Hread.
    destruct (decide ((ql, qt) = (l, t))) as [Heq | Hne].
    + inversion Heq; subst. exfalso. exact (Hretired Hread).
    + destruct (Havailable ql qt Hread)
        as [Hwrite | [v0 [Hwriteable | Hreadable]]].
      * left. exact Hwrite.
      * right. exists v0. left.
        transitivity (m !! (ql, qt)).
        -- apply lookup_insert_ne. congruence.
        -- exact Hwriteable.
      * right. exists v0. right.
        transitivity (m !! (ql, qt)).
        -- apply lookup_insert_ne. congruence.
        -- exact Hreadable.
Qed.

Lemma init_config_pending_loads_available prog :
  load_versions_produced prog ->
  pending_loads_available
    (fst (init_config prog))
    (snd (init_config prog)).
Proof.
  intros Hproduced l t Hread.
  left.
  destruct Hread as
    [i [instructions [e [op [Hlookup [Hop Hreads]]]]]].
  destruct op as [issued_at x l' t' | issued_at l' t' v].
  - cbn in Hreads. destruct Hreads as [-> ->].
    unfold init_config in Hlookup.
    cbn [fst] in Hlookup.
    apply (proj1 (lookup_fmap_Some
      (fun ops : list instr => (ops, (∅ : env)))
      prog i (instructions, e))) in Hlookup.
    destruct Hlookup as [ops [Heq Hprog]].
    inversion Heq; subst.
    destruct (Hproduced i instructions issued_at x l t Hprog Hop)
      as [j [producer [store_at [v [Hproducer Hstore]]]]].
    exists j, producer, (∅ : env), (Store store_at l t v).
    repeat split.
    + unfold init_config. cbn [fst].
      apply (proj2 (lookup_fmap_Some
        (fun ops : list instr => (ops, (∅ : env)))
        prog j (producer, (∅ : env)))).
      exists producer. split.
      * reflexivity.
      * exact Hproducer.
    + exact Hstore.
  - exact (False_rect _ Hreads).
Qed.

Lemma pending_loads_available_reachable prog np m :
  load_versions_produced prog ->
  rtc step (init_config prog) (np, m) ->
  pending_loads_available np m.
Proof.
  intros Hproduced Hrtc.
  pose proof (@rtc_ind_r config step
    (fun c => pending_loads_available (fst c) (snd c))
    (init_config prog)) as Hind.
  specialize (Hind (init_config_pending_loads_available prog Hproduced)).
  specialize (Hind ltac:(
    intros [np1 m1] [np2 m2] _ Hstep IH;
    eapply pending_loads_available_step; eauto)).
  exact (Hind (np, m) Hrtc).
Qed.

Lemma init_config_pending_stores_available prog :
  pending_stores_available
    (fst (init_config prog))
    (snd (init_config prog)).
Proof.
  unfold init_config.
  cbn.
  intros l t _.
  left. apply lookup_empty.
Qed.

Lemma pending_stores_available_reachable prog np m :
  rtc step (init_config prog) (np, m) ->
  pending_stores_available np m.
Proof.
  intros Hrtc.
  pose proof (@rtc_ind_r config step
    (fun c => pending_stores_available (fst c) (snd c))
    (init_config prog)) as Hind.
  specialize (Hind (init_config_pending_stores_available prog)).
  specialize (Hind ltac:(
    intros [np1 m1] [np2 m2] _ Hstep IH;
    eapply pending_stores_available_step; eauto)).
  exact (Hind (np, m) Hrtc).
Qed.

Lemma reachable_head_store_target_available
    prog np m i issued_at l target_at v rest e :
  rtc step (init_config prog) (np, m) ->
  np !! i = Some (Store issued_at l target_at v :: rest, e) ->
  m !! (l, target_at) = None \/
  exists v0, m !! (l, target_at) = Some (v0, Writeable).
Proof.
  intros Hrtc Hi.
  apply (pending_stores_available_reachable prog np m Hrtc).
  exists i, (Store issued_at l target_at v :: rest), e,
    (Store issued_at l target_at v).
  repeat split.
  - exact Hi.
  - apply elem_of_cons. left. reflexivity.
Qed.

Lemma store_creates_absent_version
    (np : node_pool) (m : mem)
    (i : node_id) (t1 : timestamp)
    (l : loc) (t : timestamp) (v : val)
    (rest : list instr) (e : env)
    (Hi : np !! i = Some (Store t1 l t v :: rest, e))
    (Hcheck : operation_allowed owner i (Store t1 l t v) G)
    (Habsent : m !! (l, t) = None) :
  step (np, m)
    (<[i := (rest, e)]>np, <[(l, t) := (v, Writeable)]>m).
Proof.
  eapply step_store.
  - exact Hi.
  - exact Hcheck.
  - left. exact Habsent.
Qed.

Lemma eviction_removes_version
    (np : node_pool) (m m' : mem)
    (Hevict : evict (np, m) (np, m')) :
  exists l t v,
    m !! (l, t) = Some (v, Discardable) /\
    m' !! (l, t) = None.
Proof.
  inversion Hevict; subst.
  exists l, t, v.
  split.
  - exact Hdiscardable.
  - apply lookup_delete.
Qed.

Lemma eviction_preserves_other_versions
    (m : mem) (l l' : loc) (t t' : timestamp)
    (Hdifferent : (l', t') <> (l, t)) :
  delete (l, t) m !! (l', t') = m !! (l', t').
Proof.
  apply lookup_delete_ne.
  congruence.
Qed.

Lemma eviction_preserves_observations c c' :
  evict c c' -> observationally_equivalent c c'.
Proof.
  intros Hevict.
  inversion Hevict.
  reflexivity.
Qed.

Lemma evictions_preserve_observations c c' :
  rtc evict c c' -> observationally_equivalent c c'.
Proof.
  intros Hevicts.
  induction Hevicts.
  - reflexivity.
  - unfold observationally_equivalent in *.
    transitivity (fst y).
    + exact (eviction_preserves_observations x y H).
    + exact IHHevicts.
Qed.

Lemma no_pending_write_blocks_store
    (np : node_pool) (i : node_id) (t1 : timestamp)
    (l : loc) (t : timestamp) (v : val)
    (rest : list instr) (e : env)
    (Hunused : ~ pending_write np l t)
    (Hi : np !! i = Some (Store t1 l t v :: rest, e)) :
  False.
Proof.
  apply Hunused.
  exists i, (Store t1 l t v :: rest), e, (Store t1 l t v).
  repeat split; try assumption.
  apply elem_of_cons. left. reflexivity.
Qed.

(* Once a version is Readable, one execution step preserves its value and
   either leaves it Readable or retires it as a logical Discardable tombstone. *)
Lemma readable_retired_step
    (np np' : node_pool) (m m' : mem)
    (l : loc) (t : timestamp) (v : val)
    (Hread : m !! (l, t) = Some (v, Readable))
    (Hstep : step (np, m) (np', m')) :
  m' !! (l, t) = Some (v, Readable) \/
  m' !! (l, t) = Some (v, Discardable).
Proof.
  inversion Hstep; subst.
  - (* Load: memory is unchanged. *)
    left. exact Hread.

  - (* Store. *)
    destruct (decide ((l0, t0) = (l, t))) as [Heq | Hneq].
    + inversion Heq; subst.
      destruct Hslot as [Habsent | [v_old Hwriteable]].
      * rewrite Hread in Habsent.
        discriminate.
      * rewrite Hread in Hwriteable.
        discriminate.
    + left. transitivity (m !! (l, t)).
      * apply lookup_insert_ne.
        congruence.
      * exact Hread.

  - (* Readability transition. *)
    destruct (decide ((l0, t0) = (l, t))) as [Heq | Hneq].
    + inversion Heq; subst.
      rewrite Hread in Hload.
      discriminate.
    + left. transitivity (m !! (l, t)).
      * apply lookup_insert_ne.
        congruence.
      * exact Hread.
  - (* Discardability transition. *)
    destruct (decide ((l0, t0) = (l, t))) as [Heq | Hneq].
    + inversion Heq; subst.
      rewrite Hread in Hload.
      injection Hload as ->.
      right. apply lookup_insert.
    + left. transitivity (m !! (l, t)).
      * apply lookup_insert_ne.
        congruence.
      * exact Hread.
Qed.

Lemma discardable_preserved_step
    (np np' : node_pool) (m m' : mem)
    (l : loc) (t : timestamp) (v : val)
    (Hdiscarded : m !! (l, t) = Some (v, Discardable))
    (Hstep : step (np, m) (np', m')) :
  m' !! (l, t) = Some (v, Discardable).
Proof.
  inversion Hstep; subst.
  - exact Hdiscarded.
  - destruct (decide ((l0, t0) = (l, t))) as [Heq | Hneq].
    + inversion Heq; subst.
      destruct Hslot as [Habsent | [v_old Hwriteable]].
      * rewrite Hdiscarded in Habsent. discriminate.
      * rewrite Hdiscarded in Hwriteable. discriminate.
    + transitivity (m !! (l, t)).
      * apply lookup_insert_ne. congruence.
      * exact Hdiscarded.
  - destruct (decide ((l0, t0) = (l, t))) as [Heq | Hneq].
    + inversion Heq; subst.
      rewrite Hdiscarded in Hload. discriminate.
    + transitivity (m !! (l, t)).
      * apply lookup_insert_ne. congruence.
      * exact Hdiscarded.
  - destruct (decide ((l0, t0) = (l, t))) as [Heq | Hneq].
    + inversion Heq; subst.
      rewrite Hdiscarded in Hload. discriminate.
    + transitivity (m !! (l, t)).
      * apply lookup_insert_ne. congruence.
      * exact Hdiscarded.
Qed.

Definition retired_version
    (m : mem) (l : loc) (t : timestamp) (v : val) : Prop :=
  m !! (l, t) = Some (v, Readable) \/
  m !! (l, t) = Some (v, Discardable).

Lemma retired_version_step
    (np np' : node_pool) (m m' : mem)
    (l : loc) (t : timestamp) (v : val)
    (Hretired : retired_version m l t v)
    (Hstep : step (np, m) (np', m')) :
  retired_version m' l t v.
Proof.
  destruct Hretired as [Hread | Hdiscarded].
  - exact (readable_retired_step np np' m m' l t v Hread Hstep).
  - right. exact (discardable_preserved_step np np' m m' l t v Hdiscarded Hstep).
Qed.

Lemma retired_version_reachable
    (np np' : node_pool) (m m' : mem)
    (l : loc) (t : timestamp) (v : val)
    (Hretired : retired_version m l t v)
    (Hrtc : rtc step (np, m) (np', m')) :
  retired_version m' l t v.
Proof.
  pose proof (@rtc_ind_r config step
    (fun c => retired_version (snd c) l t v) (np, m)) as Hind.
  specialize (Hind Hretired).
  specialize (Hind ltac:(
    intros [np1 m1] [np2 m2] _ Hstep IH;
    eapply retired_version_step; eauto)).
  exact (Hind (np', m') Hrtc).
Qed.

Definition location_of (op : instr) : loc :=
  match op with
  | Load _ _ l _ => l
  | Store _ l _ _ => l
  end.

(* Every permitted remote operation targets a graph neighbor. *)
Lemma permitted_remote_operation_has_edge
    (issuer : node_id) (op : instr)
    (Hremote : issuer <> owner (location_of op))
    (Hallowed : operation_allowed owner issuer op G) :
  neighbors G issuer (owner (location_of op)).
Proof.
  destruct op as [t1 x l t | t1 l t v];
    simpl in Hremote, Hallowed |-;
    destruct Hallowed as [Htarget _];
    destruct Htarget as [Hlocal | Hneighbor].
  - contradiction.
  - exact Hneighbor.
  - contradiction.
  - exact Hneighbor.
Qed.

(* A permitted remote load reads a version at least one readability lag
   behind its issuing timestamp. *)
Lemma remote_load_targets_old_version
    (issuer : node_id) (issued_at : timestamp)
    (x : var) (l : loc) (target_at : timestamp)
    (Hremote : issuer <> owner l)
    (Hallowed :
       operation_allowed owner issuer
         (Load issued_at x l target_at) G) :
  exists readable discardable,
    readable_lag G issuer (owner l) = Some readable /\
    discard_lag G issuer (owner l) = Some discardable /\
    target_at + readable <= issued_at /\
    issued_at <= target_at + discardable /\
    target_at < issued_at.
Proof.
  simpl in Hallowed.
  destruct Hallowed as [_ Htime].

  assert (Heqb : Nat.eqb (owner l) issuer = false).
  {
    apply Nat.eqb_neq.
    congruence.
  }

  rewrite Heqb in Htime.
  destruct Htime as
    [readable [discardable [Hreadable [Hdiscardable [Hlower Hupper]]]]].
  exists readable, discardable.
  repeat split; try assumption.

  pose proof
    (channel_positive G issuer (owner l) readable Hreadable)
    as Hpositive.
  lia.
Qed.

(* A permitted remote store writes a version at least one readability lag
   ahead of its issuing timestamp. *)
Lemma remote_store_targets_future_version
    (issuer : node_id) (issued_at : timestamp)
    (l : loc) (target_at : timestamp) (v : val)
    (Hremote : issuer <> owner l)
    (Hallowed :
       operation_allowed owner issuer
         (Store issued_at l target_at v) G) :
  exists d,
    readable_lag G issuer (owner l) = Some d /\
    issued_at + d <= target_at /\
    issued_at < target_at.
Proof.
  simpl in Hallowed.
  destruct Hallowed as [_ Htime].

  assert (Heqb : Nat.eqb (owner l) issuer = false).
  {
    apply Nat.eqb_neq.
    congruence.
  }

  rewrite Heqb in Htime.
  destruct Htime as [d [Hlag Hbound]].
  exists d.
  repeat split; try assumption.

  pose proof (channel_positive G issuer (owner l) d Hlag) as Hpositive.
  lia.
Qed.

(* A readable memory version has a unique value, so any two loads
   observing that version obtain the same result. *)
Lemma readable_version_has_unique_value
    (m : mem) (l : loc) (t : timestamp) (v1 v2 : val)
    (Hload1 : m !! (l, t) = Some (v1, Readable))
    (Hload2 : m !! (l, t) = Some (v2, Readable)) :
  v1 = v2.
Proof.
  rewrite Hload1 in Hload2.
  inversion Hload2.
  reflexivity.
Qed.

Definition finished (c : config) := map_Forall (λ _ '(is, _), is = []) (fst c).

Lemma eviction_preserves_finished c c' :
  evict c c' -> (finished c <-> finished c').
Proof.
  intros Hevict.
  inversion Hevict.
  reflexivity.
Qed.

(* This would also be an interesting property to prove on the status_semantics, which uses a more concrete
   scheduling strategy. *)
Definition deadlock_free (init : config) :=
  forall state, rtc step init state -> finished state \/ exists state', step state state'.

End semantics.

(* Examples 

Ring(n):
for some number of timestamps t:
  at t, store (n + 1) (t + 1) (t + 1)
  at t, load n t , make sure the value is t

*)
Definition ring_program (n i : nat) : list instr :=
  [Store 0 ((i + 1) `mod` n) 1 1%Z;
   Store 1 ((i + 1) `mod` n) 2 2%Z;
   Load 1 "x"%string i 1].

Definition ring (n : nat) : gmap node_id (list instr) :=
  foldr (fun i prog => <[i := ring_program n i]> prog)
    ∅ (seq 0 n).

(* ring_lookup proves:
    if node i is part of a ring of size n, looking up i in the ring program returns exactly
    its expected instructions
*)

Lemma foldr_insert_lookup_present
    {A : Type} (f : nat -> A) (xs : list nat) i :
  i ∈ xs ->
  (foldr
     (fun j (m : gmap nat A) => <[j := f j]> m)
     ∅ xs) !! i = Some (f i).
Proof.
  induction xs as [|j xs IH].
  - intros Hi.
    rewrite elem_of_nil in Hi.
    contradiction.
  - intros Hi.
    rewrite elem_of_cons in Hi.
    destruct Hi as [Hij | Hi].
    + subst j.
      cbn.
      rewrite lookup_insert_eq.
      reflexivity.
    + cbn.
      destruct (decide (j = i)) as [-> | Hne].
      * rewrite lookup_insert_eq.
        reflexivity.
      * rewrite lookup_insert_ne; last exact Hne.
        apply IH.
        exact Hi.
Qed.

Lemma ring_lookup n i :
  i < n ->
  ring n !! i = Some (ring_program n i).
Proof.
  intros Hi.
  unfold ring.
  refine (@foldr_insert_lookup_present
    (list instr) (ring_program n) (seq 0 n) i _).
  apply elem_of_seq.
  lia.
Qed.

(* Every key occurring in a finite map built by folding insert over a
   list comes from that list. *)
Lemma foldr_insert_lookup_member
    {A : Type} (f : nat -> A) (xs : list nat) i x :
  (foldr
     (fun j (m : gmap nat A) => <[j := f j]> m)
     ∅ xs) !! i = Some x ->
  i ∈ xs.
Proof.
  induction xs as [|j xs IH].
  - intros Hlookup.
    cbn in Hlookup.
    rewrite lookup_empty in Hlookup.
    discriminate.
  - intros Hlookup.
    cbn in Hlookup.
    rewrite elem_of_cons.
    destruct (Nat.eq_dec j i) as [-> | Hne].
    + left.
      reflexivity.
    + right.
      apply IH.
      transitivity
        ((<[j := f j]>
            (foldr
              (fun j0 (m : gmap nat A) => <[j0 := f j0]> m)
              ∅ xs)) !! i).
      * symmetry.
        apply lookup_insert_ne.
        exact Hne.
      * exact Hlookup.
Qed.

Lemma ring_lookup_in_bounds n i instructions :
  ring n !! i = Some instructions ->
  i < n.
Proof.
  intros Hlookup.
  assert (Hmember : i ∈ seq 0 n).
  {
    apply
      (foldr_insert_lookup_member
        (ring_program n) (seq 0 n) i instructions).
    exact Hlookup.
  }
  apply (proj1 (elem_of_seq 0 n i)) in Hmember.
  lia.
Qed.

Definition neighbor n n1 n2 : bool := (n2 =? (n1 + 1) `mod` n) || (n2 =? (n1 + n - 1) `mod` n).

Program Definition ring_graph n : comm_graph := {|
  world_size := n;
  readable_lag n1 n2 := 
    if decide (n1 < n /\ n2 < n /\ n1 <> n2 /\
          neighbor n n1 n2 = true) then Some 1 else None;
  discard_lag n1 n2 :=
    if decide (n1 < n /\ n2 < n /\ n1 <> n2 /\
          neighbor n n1 n2 = true) then Some 1 else None
|}.

(* channel_in_bounds *)
Next Obligation.
  intros n p q d Hlag.
  cbn in Hlag.
  destruct (decide
    (p < n /\ q < n /\ p <> q /\ neighbor n p q = true))
    as [Hedge | Hnoedge].
  - destruct Hedge as [Hp [Hq _]].
    split; assumption.
  - discriminate.
Qed.

(* channel_irreflexive *)
Next Obligation.
  intros n p Hp.
  cbn.
  case_decide; naive_solver.
Qed.

(* channel_positive *)
Next Obligation.
  intros n p q d Hlag.
  cbn in Hlag.
  destruct (decide
    (p < n /\ q < n /\ p <> q /\ neighbor n p q = true)).
  - injection Hlag as <-.
    lia.
  - discriminate Hlag.
Qed.

(* discard_channel *)
Next Obligation.
  intros n p q.
  cbn.
  reflexivity.
Qed.

(* channel_discard_bounds *)
Next Obligation.
  intros n p q readable discardable Hreadable Hdiscardable.
  cbn in Hreadable, Hdiscardable.
  destruct (decide
    (p < n /\ q < n /\ p <> q /\ neighbor n p q = true)).
  - injection Hreadable as <-.
    injection Hdiscardable as <-.
    lia.
  - discriminate Hreadable.
Qed.

(* A non-local successor in the ring is connected by an edge with lag 1. *)
Lemma ring_successor_lag n i :
  i < n ->
  (i + 1) `mod` n <> i ->
  readable_lag (ring_graph n) i ((i + 1) `mod` n) = Some 1.
Proof.
  intros Hi Hneq.
  cbn.
  rewrite decide_True.
  - reflexivity.
  - repeat split.
    + exact Hi.
    + apply Nat.mod_upper_bound.
      lia.
    + lia.
    + unfold neighbor.
      rewrite Nat.eqb_refl.
      reflexivity.
Qed.

(* A ring store is permitted: it is either local, or targets the
   next ring neighbor at least one timestamp in the future. *)
Lemma ring_store_allowed n i issued_at target_at v :
  i < n ->
  issued_at + 1 <= target_at ->
  operation_allowed
    id i
    (Store issued_at ((i + 1) `mod` n) target_at v)
    (ring_graph n).
Proof.
  intros Hi Htime.
  unfold operation_allowed.
  cbn.

  destruct (Nat.eq_dec ((i + 1) `mod` n) i) as [Heq | Hneq].
  - rewrite Heq, Nat.eqb_refl.
    split.
    + left. reflexivity.
    + lia.
  - assert
      (Hlag :
         readable_lag (ring_graph n) i ((i + 1) `mod` n) = Some 1).
    {
      apply ring_successor_lag; assumption.
    }
    assert (Heqb : ((i + 1) `mod` n =? i) = false).
    {
      apply Nat.eqb_neq.
      exact Hneq.
    }
    rewrite Heqb.
    split.
    + right. exists 1. exact Hlag.
    + exists 1. split.
      * exact Hlag.
      * lia.
Qed.

(* Every node's first store is permitted at timestamp 0,
   targeting its successor's version at timestamp 1. *)
Lemma ring_first_store_allowed n i :
  i < n ->
  operation_allowed
    id i
    (Store 0 ((i + 1) `mod` n) 1 1%Z)
    (ring_graph n).
Proof.
  intros Hi.
  apply ring_store_allowed; lia.
Qed.

(* Every node's second store is permitted at timestamp 1,
   targeting its successor's version at timestamp 2. *)
Lemma ring_second_store_allowed n i :
  i < n ->
  operation_allowed
    id i
    (Store 1 ((i + 1) `mod` n) 2 2%Z)
    (ring_graph n).
Proof.
  intros Hi.
  apply ring_store_allowed; lia.
Qed.

(* Every node's final load is permitted because it reads its own
   location at the current timestamp. *)
Lemma ring_load_allowed n i :
  operation_allowed
    id i
    (Load 1 "x"%string i 1)
    (ring_graph n).
Proof.
  unfold operation_allowed.
  cbn.
  split.
  - left. reflexivity.
  - rewrite Nat.eqb_refl.
    lia.
Qed.

(* Every in-bounds node initially has its complete ring program
   and an empty environment. *)
Lemma init_ring_lookup n i :
  i < n ->
  fst (init_config (ring n)) !! i =
    Some (ring_program n i, ∅).
Proof.
  intros Hi.
  unfold init_config.
  cbn [fst].
  apply (proj2 (lookup_fmap_Some
    (λ is : list instr, (is, (∅ : env)))
    (ring n) i
    (ring_program n i, (∅ : env)))).
  exists (ring_program n i).
  split.
  - reflexivity.
  - exact (ring_lookup n i Hi).
Qed.

(* The initial node pool has no nodes outside the ring. *)
Lemma init_ring_lookup_in_bounds n i p :
  fst (init_config (ring n)) !! i = Some p ->
  i < n.
Proof.
  intros Hlookup.
  unfold init_config in Hlookup.
  cbn [fst] in Hlookup.
  apply (proj1 (lookup_fmap_Some
    (λ is : list instr, (is, (∅ : env)))
    (ring n) i p)) in Hlookup.
  destruct Hlookup as [instructions [_ Hring]].
  eapply ring_lookup_in_bounds.
  exact Hring.
Qed.

(* Node pool obtained after nodes 0 through k-1 have executed their
   first stores. *)
Fixpoint ring_first_np (n k : nat) : node_pool :=
  match k with
  | 0 => fst (init_config (ring n))
  | S k' =>
      <[k' :=
        ([Store 1 ((k' + 1) `mod` n) 2 2%Z;
          Load 1 "x"%string k' 1], (∅ : env))]>
        (ring_first_np n k')
  end.

(* Memory obtained after nodes 0 through k-1 have executed their
   first stores. *)
Fixpoint ring_first_mem (n k : nat) : mem :=
  match k with
  | 0 => ∅
  | S k' =>
      <[((k' + 1) `mod` n, 1) := (1%Z, Writeable)]>
        (ring_first_mem n k')
  end.

Lemma ring_first_np_0 n :
  ring_first_np n 0 =
    fst (init_config (ring n)).
Proof.
  reflexivity.
Qed.

Lemma ring_first_np_S n k :
  ring_first_np n (S k) =
    <[k :=
      ([Store 1 ((k + 1) `mod` n) 2 2%Z;
        Load 1 "x"%string k 1], (∅ : env))]>
      (ring_first_np n k).
Proof.
  reflexivity.
Qed.

Lemma ring_first_mem_0 n :
  ring_first_mem n 0 = (∅ : mem).
Proof.
  reflexivity.
Qed.

Lemma ring_first_mem_S n k :
  ring_first_mem n (S k) =
    <[((k + 1) `mod` n, 1) := (1%Z, Writeable)]>
      (ring_first_mem n k).
Proof.
  reflexivity.
Qed.

(* A node not yet visited by the first-store sweep still has its
   complete initial program. *)
Lemma ring_first_np_unprocessed n k i :
  k <= i ->
  i < n ->
  ring_first_np n k !! i =
    Some (ring_program n i, ∅).
Proof.
  intros Hki Hin.
  induction k as [|k IH].
  - rewrite ring_first_np_0.
    apply init_ring_lookup.
    exact Hin.
  - rewrite ring_first_np_S.
    transitivity (ring_first_np n k !! i).
    + apply lookup_insert_ne.
      lia.
    + apply IH.
      lia.
Qed.

Lemma ring_first_np_lookup_in_bounds n k i p :
  k <= n ->
  ring_first_np n k !! i = Some p ->
  i < n.
Proof.
  intros Hkn.
  induction k as [|k IH].
  - rewrite ring_first_np_0.
    apply init_ring_lookup_in_bounds.
  - intros Hlookup.
    rewrite ring_first_np_S in Hlookup.
    destruct (Nat.eq_dec k i) as [-> | Hne].
    + lia.
    + apply IH.
      * lia.
      * transitivity
          ((<[k :=
              ([Store 1 ((k + 1) `mod` n) 2 2%Z;
                Load 1 "x"%string k 1], (∅ : env))]>
             (ring_first_np n k)) !! i).
        -- symmetry.
           apply lookup_insert_ne.
           exact Hne.
        -- exact Hlookup.
Qed.

(* A node already visited by the first-store sweep has only its
   second store and final load remaining. *)
Lemma ring_first_np_processed n k i :
  i < k ->
  k <= n ->
  ring_first_np n k !! i =
    Some
      ([Store 1 ((i + 1) `mod` n) 2 2%Z;
        Load 1 "x"%string i 1], ∅).
Proof.
  intros Hik Hkn.
  induction k as [|k IH].
  - lia.
  - rewrite ring_first_np_S.
    destruct (Nat.eq_dec i k) as [-> | Hne].
    + apply lookup_insert_eq.
    + transitivity (ring_first_np n k !! i).
      * apply lookup_insert_ne.
        lia.
      * apply IH; lia.
Qed.

(* Successor modulo n is injective on the bounded node identifiers. *)
Lemma ring_successor_inj n i j :
  i < n ->
  j < n ->
  (i + 1) `mod` n = (j + 1) `mod` n ->
  i = j.
Proof.
  intros Hi Hj Heq.
  assert (Hn : 0 < n) by lia.
  assert (Hn0 : n <> 0) by lia.
  replace (i + 1) with (S i) in Heq by lia.
  replace (j + 1) with (S j) in Heq by lia.
  destruct (Nat.eq_dec (S i) n) as [Hi_last | Hi_not_last].
  - destruct (Nat.eq_dec (S j) n) as [Hj_last | Hj_not_last].
    + lia.
    + assert (Hj_small : S j < n) by lia.
      rewrite Hi_last in Heq.
      rewrite Nat.mod_same in Heq by exact Hn0.
      rewrite Nat.mod_small in Heq by exact Hj_small.
      lia.
  - assert (Hi_small : S i < n) by lia.
    destruct (Nat.eq_dec (S j) n) as [Hj_last | Hj_not_last].
    + rewrite Nat.mod_small in Heq by exact Hi_small.
      rewrite Hj_last in Heq.
      rewrite Nat.mod_same in Heq by exact Hn0.
      lia.
    + assert (Hj_small : S j < n) by lia.
      rewrite Nat.mod_small in Heq by exact Hi_small.
      rewrite Nat.mod_small in Heq by exact Hj_small.
      lia.
Qed.

Definition ring_predecessor (n i : nat) : nat :=
  if Nat.eq_dec i 0 then n - 1 else i - 1.

(* Every bounded location is the successor of a bounded ring node. *)
Lemma ring_predecessor_spec n i :
  i < n ->
  ring_predecessor n i < n /\
  (ring_predecessor n i + 1) `mod` n = i.
Proof.
  intros Hin.
  unfold ring_predecessor.
  destruct (Nat.eq_dec i 0) as [-> | Hne].
  - split.
    + lia.
    + replace (n - 1 + 1) with n by lia.
      rewrite Nat.mod_same by lia.
      reflexivity.
  - split.
    + lia.
    + replace (i - 1 + 1) with i by lia.
      apply Nat.mod_small.
      exact Hin.
Qed.

Lemma ring_load_versions_produced n :
  load_versions_produced (ring n).
Proof.
  intros i instructions issued_at x l t Hlookup Hload.
  pose proof (ring_lookup_in_bounds n i instructions Hlookup) as Hin.
  pose proof (ring_lookup n i Hin) as Hexpected.
  pose proof (eq_trans (eq_sym Hlookup) Hexpected) as Heq.
  injection Heq as ->.
  unfold ring_program in Hload.
  repeat rewrite elem_of_cons in Hload.
  rewrite elem_of_nil in Hload.
  destruct Hload as [Hbad | [Hbad | [Hload | Hbad]]].
  - discriminate.
  - discriminate.
  - inversion Hload; subst.
    destruct (ring_predecessor_spec n i Hin) as [Hpred Hsuccessor].
    exists (ring_predecessor n i),
      (ring_program n (ring_predecessor n i)), 0, 1%Z.
    split.
    + apply ring_lookup. exact Hpred.
    + unfold ring_program.
      apply elem_of_cons. left. rewrite Hsuccessor. reflexivity.
  - contradiction.
Qed.

(* No first-store target belonging to an unprocessed node has yet
   been allocated. *)
Lemma ring_first_mem_unprocessed n k i :
  k <= i ->
  i < n ->
  ring_first_mem n k !! (((i + 1) `mod` n), 1) = None.
Proof.
  intros Hki Hin.
  induction k as [|k IH].
  - rewrite ring_first_mem_0.
    apply lookup_empty.
  - rewrite ring_first_mem_S.
    transitivity
      (ring_first_mem n k !! (((i + 1) `mod` n), 1)).
    + apply lookup_insert_ne.
      intro Hkeys.
      apply (f_equal fst) in Hkeys.
      cbn in Hkeys.
      assert (Hkn : k < n) by lia.
      pose proof
        (ring_successor_inj n k i Hkn Hin Hkeys)
        as Hbad.
      lia.
    + apply IH.
      lia.
Qed.

(* Immediately before node k's first store, its target slot is absent. *)
Lemma ring_first_mem_fresh n k :
  k < n ->
  ring_first_mem n k !! (((k + 1) `mod` n), 1) = None.
Proof.
  intros Hkn.
  apply ring_first_mem_unprocessed.
  - lia.
  - exact Hkn.
Qed.

(* Every processed node has written its timestamp-1 successor slot. *)
Lemma ring_first_mem_processed n k i :
  i < k ->
  k <= n ->
  ring_first_mem n k !! (((i + 1) `mod` n), 1) =
    Some (1%Z, Writeable).
Proof.
  intros Hik Hkn.
  induction k as [|k IH].
  - lia.
  - rewrite ring_first_mem_S.
    destruct (Nat.eq_dec i k) as [-> | Hne].
    + apply lookup_insert_eq.
    + transitivity
        (ring_first_mem n k !! (((i + 1) `mod` n), 1)).
      * apply lookup_insert_ne.
        intro Hkeys.
        apply (f_equal fst) in Hkeys.
        cbn in Hkeys.
        assert (Hin : i < n) by lia.
        assert (Hkn' : k < n) by lia.
        pose proof
          (ring_successor_inj n k i Hkn' Hin Hkeys)
          as Heq.
        apply Hne.
        symmetry.
        exact Heq.
      * apply IH; lia.
Qed.

(* The first-store sweep only allocates timestamp-1 versions. *)
Lemma ring_first_mem_timestamp2_absent n k l :
  ring_first_mem n k !! (l, 2) = None.
Proof.
  induction k as [|k IH].
  - rewrite ring_first_mem_0.
    apply lookup_empty.
  - rewrite ring_first_mem_S.
    transitivity (ring_first_mem n k !! (l, 2)).
    + apply lookup_insert_ne.
      intro Hkeys.
      apply (f_equal snd) in Hkeys.
      discriminate.
    + exact IH.
Qed.

Lemma ring_first_step n k :
  k < n ->
  step id (ring_graph n)
    (ring_first_np n k, ring_first_mem n k)
    (ring_first_np n (S k), ring_first_mem n (S k)).
Proof.
  intros Hkn.
  pose proof
    (ring_first_np_unprocessed n k k (Nat.le_refl k) Hkn)
    as Hlookup.
  unfold ring_program in Hlookup.
  rewrite ring_first_np_S.
  rewrite ring_first_mem_S.
  eapply step_store.
  - exact Hlookup.
  - apply ring_first_store_allowed.
    exact Hkn.
  - left.
    apply ring_first_mem_fresh.
    exact Hkn.
Qed.

Lemma ring_first_sweep n k :
  k <= n ->
  rtc (step id (ring_graph n))
    (init_config (ring n))
    (ring_first_np n k, ring_first_mem n k).
Proof.
  intros Hkn.
  induction k as [|k IH].
  - apply rtc_refl.
  - eapply rtc_r.
    + apply IH.
      lia.
    + apply ring_first_step.
      lia.
Qed.

(* Node pool obtained after all first stores and the first k second
   stores have executed. *)
Fixpoint ring_second_np (n k : nat) : node_pool :=
  match k with
  | 0 => ring_first_np n n
  | S k' =>
      <[k' :=
        ([Load 1 "x"%string k' 1], (∅ : env))]>
        (ring_second_np n k')
  end.

(* Memory obtained after all first stores and the first k second
   stores have executed. *)
Fixpoint ring_second_mem (n k : nat) : mem :=
  match k with
  | 0 => ring_first_mem n n
  | S k' =>
      <[((k' + 1) `mod` n, 2) := (2%Z, Writeable)]>
        (ring_second_mem n k')
  end.

Lemma ring_second_np_0 n :
  ring_second_np n 0 = ring_first_np n n.
Proof.
  reflexivity.
Qed.

Lemma ring_second_np_S n k :
  ring_second_np n (S k) =
    <[k :=
      ([Load 1 "x"%string k 1], (∅ : env))]>
      (ring_second_np n k).
Proof.
  reflexivity.
Qed.

Lemma ring_second_mem_0 n :
  ring_second_mem n 0 = ring_first_mem n n.
Proof.
  reflexivity.
Qed.

Lemma ring_second_mem_S n k :
  ring_second_mem n (S k) =
    <[((k + 1) `mod` n, 2) := (2%Z, Writeable)]>
      (ring_second_mem n k).
Proof.
  reflexivity.
Qed.

(* A node not yet visited by the second-store sweep still has its
   second store followed by its final load. *)
Lemma ring_second_np_unprocessed n k i :
  k <= i ->
  i < n ->
  ring_second_np n k !! i =
    Some
      ([Store 1 ((i + 1) `mod` n) 2 2%Z;
        Load 1 "x"%string i 1], ∅).
Proof.
  intros Hki Hin.
  induction k as [|k IH].
  - rewrite ring_second_np_0.
    apply ring_first_np_processed.
    + exact Hin.
    + lia.
  - rewrite ring_second_np_S.
    transitivity (ring_second_np n k !! i).
    + apply lookup_insert_ne.
      lia.
    + apply IH.
      lia.
Qed.

(* A node already visited by the second-store sweep has only its
   final load remaining. *)
Lemma ring_second_np_processed n k i :
  i < k ->
  k <= n ->
  ring_second_np n k !! i =
    Some ([Load 1 "x"%string i 1], ∅).
Proof.
  intros Hik Hkn.
  induction k as [|k IH].
  - lia.
  - rewrite ring_second_np_S.
    destruct (Nat.eq_dec i k) as [-> | Hne].
    + apply lookup_insert_eq.
    + transitivity (ring_second_np n k !! i).
      * apply lookup_insert_ne.
        lia.
      * apply IH; lia.
Qed.

Lemma ring_second_np_lookup_in_bounds n k i p :
  k <= n ->
  ring_second_np n k !! i = Some p ->
  i < n.
Proof.
  intros Hkn.
  induction k as [|k IH].
  - rewrite ring_second_np_0.
    apply ring_first_np_lookup_in_bounds.
    lia.
  - intros Hlookup.
    rewrite ring_second_np_S in Hlookup.
    destruct (Nat.eq_dec k i) as [-> | Hne].
    + lia.
    + apply IH.
      * lia.
      * transitivity
          ((<[k :=
              ([Load 1 "x"%string k 1], (∅ : env))]>
             (ring_second_np n k)) !! i).
        -- symmetry.
           apply lookup_insert_ne.
           exact Hne.
        -- exact Hlookup.
Qed.

(* After all second stores, every remaining program is its node's
   singleton final load. *)
Lemma ring_second_np_final_lookup n i instructions e :
  ring_second_np n n !! i = Some (instructions, e) ->
  instructions = [Load 1 "x"%string i 1] /\ e = ∅.
Proof.
  intros Hlookup.
  assert (Hin : i < n).
  {
    eapply ring_second_np_lookup_in_bounds.
    - exact (Nat.le_refl n).
    - exact Hlookup.
  }
  pose proof
    (ring_second_np_processed n n i Hin (Nat.le_refl n))
    as Hexpected.
  pose proof (eq_trans (eq_sym Hlookup) Hexpected) as Heq.
  inversion Heq.
  split; reflexivity.
Qed.

(* Timestamp-2 targets belonging to unprocessed nodes have not yet
   been allocated by the second-store sweep. *)
Lemma ring_second_mem_unprocessed n k i :
  k <= i ->
  i < n ->
  ring_second_mem n k !! (((i + 1) `mod` n), 2) = None.
Proof.
  intros Hki Hin.
  induction k as [|k IH].
  - rewrite ring_second_mem_0.
    apply ring_first_mem_timestamp2_absent.
  - rewrite ring_second_mem_S.
    transitivity
      (ring_second_mem n k !! (((i + 1) `mod` n), 2)).
    + apply lookup_insert_ne.
      intro Hkeys.
      apply (f_equal fst) in Hkeys.
      cbn in Hkeys.
      assert (Hkn : k < n) by lia.
      pose proof
        (ring_successor_inj n k i Hkn Hin Hkeys)
        as Hbad.
      lia.
    + apply IH.
      lia.
Qed.

Lemma ring_second_mem_fresh n k :
  k < n ->
  ring_second_mem n k !! (((k + 1) `mod` n), 2) = None.
Proof.
  intros Hkn.
  apply ring_second_mem_unprocessed.
  - lia.
  - exact Hkn.
Qed.

(* Second stores only modify timestamp-2 slots, so every completed
   first-store entry remains Writeable. *)
Lemma ring_second_mem_timestamp1 n k i :
  i < n ->
  ring_second_mem n k !! (((i + 1) `mod` n), 1) =
    Some (1%Z, Writeable).
Proof.
  intros Hin.
  induction k as [|k IH].
  - rewrite ring_second_mem_0.
    apply ring_first_mem_processed.
    + exact Hin.
    + exact (Nat.le_refl n).
  - rewrite ring_second_mem_S.
    transitivity
      (ring_second_mem n k !! (((i + 1) `mod` n), 1)).
    + apply lookup_insert_ne.
      intro Hkeys.
      apply (f_equal snd) in Hkeys.
      discriminate.
    + exact IH.
Qed.

Lemma ring_second_mem_location_timestamp1 n k i :
  i < n ->
  ring_second_mem n k !! (i, 1) =
    Some (1%Z, Writeable).
Proof.
  intros Hin.
  destruct (ring_predecessor_spec n i Hin) as [Hpred Hsuccessor].
  transitivity
    (ring_second_mem n k !!
      (((ring_predecessor n i + 1) `mod` n), 1)).
  - rewrite Hsuccessor.
    reflexivity.
  - apply ring_second_mem_timestamp1.
    exact Hpred.
Qed.

Lemma ring_second_step n k :
  k < n ->
  step id (ring_graph n)
    (ring_second_np n k, ring_second_mem n k)
    (ring_second_np n (S k), ring_second_mem n (S k)).
Proof.
  intros Hkn.
  pose proof
    (ring_second_np_unprocessed n k k (Nat.le_refl k) Hkn)
    as Hlookup.
  rewrite ring_second_np_S.
  rewrite ring_second_mem_S.
  eapply step_store.
  - exact Hlookup.
  - apply ring_second_store_allowed.
    exact Hkn.
  - left.
    apply ring_second_mem_fresh.
    exact Hkn.
Qed.

Lemma ring_second_sweep n k :
  k <= n ->
  rtc (step id (ring_graph n))
    (ring_first_np n n, ring_first_mem n n)
    (ring_second_np n k, ring_second_mem n k).
Proof.
  intros Hkn.
  induction k as [|k IH].
  - apply rtc_refl.
  - eapply rtc_r.
    + apply IH.
      lia.
    + apply ring_second_step.
      lia.
Qed.

(* The first two transition sweeps compose. *)
Lemma ring_store_sweeps n :
  rtc (step id (ring_graph n))
    (init_config (ring n))
    (ring_second_np n n, ring_second_mem n n).
Proof.
  etransitivity.
  - apply ring_first_sweep.
    exact (Nat.le_refl n).
  - apply ring_second_sweep.
    exact (Nat.le_refl n).
Qed.

(* Once both store sweeps finish, every remaining instruction is a
   load, so no timestamp-1 version has a pending writer. *)
Lemma ring_second_np_visible n l :
  visible (ring_second_np n n) l 1.
Proof.
  unfold visible, pending_write.
  intros Hpending.
  destruct Hpending as
    [i [instructions [e [op [Hlookup [Hop Hwrites]]]]]].
  destruct
    (ring_second_np_final_lookup n i instructions e Hlookup)
    as [-> ->].
  rewrite elem_of_cons in Hop.
  destruct Hop as [Hop | Hop].
  - subst op.
    cbn in Hwrites.
    exact Hwrites.
  - rewrite elem_of_nil in Hop.
    contradiction.
Qed.

(* Memory after locations 0 through k-1 have had their timestamp-1
   versions marked Readable. *)
Fixpoint ring_readable_mem (n k : nat) : mem :=
  match k with
  | 0 => ring_second_mem n n
  | S k' =>
      <[(k', 1) := (1%Z, Readable)]>
        (ring_readable_mem n k')
  end.

Lemma ring_readable_mem_0 n :
  ring_readable_mem n 0 = ring_second_mem n n.
Proof.
  reflexivity.
Qed.

Lemma ring_readable_mem_S n k :
  ring_readable_mem n (S k) =
    <[(k, 1) := (1%Z, Readable)]>
      (ring_readable_mem n k).
Proof.
  reflexivity.
Qed.

(* A location not yet visited by the readability sweep remains
   Writeable at timestamp 1. *)
Lemma ring_readable_mem_unprocessed n k i :
  k <= i ->
  i < n ->
  ring_readable_mem n k !! (i, 1) =
    Some (1%Z, Writeable).
Proof.
  intros Hki Hin.
  induction k as [|k IH].
  - rewrite ring_readable_mem_0.
    apply ring_second_mem_location_timestamp1.
    exact Hin.
  - rewrite ring_readable_mem_S.
    transitivity (ring_readable_mem n k !! (i, 1)).
    + apply lookup_insert_ne.
      intro Hkeys.
      apply (f_equal fst) in Hkeys.
      cbn in Hkeys.
      lia.
    + apply IH.
      lia.
Qed.

Lemma ring_readable_step n k :
  k < n ->
  step id (ring_graph n)
    (ring_second_np n n, ring_readable_mem n k)
    (ring_second_np n n, ring_readable_mem n (S k)).
Proof.
  intros Hkn.
  rewrite ring_readable_mem_S.
  eapply step_readable.
  - apply ring_readable_mem_unprocessed.
    + exact (Nat.le_refl k).
    + exact Hkn.
  - apply ring_second_np_visible.
Qed.

Lemma ring_readable_sweep n k :
  k <= n ->
  rtc (step id (ring_graph n))
    (ring_second_np n n, ring_second_mem n n)
    (ring_second_np n n, ring_readable_mem n k).
Proof.
  intros Hkn.
  induction k as [|k IH].
  - apply rtc_refl.
  - eapply rtc_r.
    + apply IH.
      lia.
    + apply ring_readable_step.
      lia.
Qed.

(* The two store sweeps followed by the readability sweep. *)
Lemma ring_store_readable_sweeps n :
  rtc (step id (ring_graph n))
    (init_config (ring n))
    (ring_second_np n n, ring_readable_mem n n).
Proof.
  etransitivity.
  - apply ring_store_sweeps.
  - apply ring_readable_sweep.
    exact (Nat.le_refl n).
Qed.

Lemma ring_readable_mem_processed n k i :
  i < k ->
  k <= n ->
  ring_readable_mem n k !! (i, 1) =
    Some (1%Z, Readable).
Proof.
  intros Hik Hkn.
  induction k as [|k IH].
  - lia.
  - rewrite ring_readable_mem_S.
    destruct (Nat.eq_dec i k) as [-> | Hne].
    + apply lookup_insert_eq.
    + transitivity (ring_readable_mem n k !! (i, 1)).
      * apply lookup_insert_ne.
        intro Hkeys.
        apply (f_equal fst) in Hkeys.
        cbn in Hkeys.
        exact (Hne (eq_sym Hkeys)).
      * apply IH; lia.
Qed.

Lemma ring_readable_mem_final n i :
  i < n ->
  ring_readable_mem n n !! (i, 1) =
    Some (1%Z, Readable).
Proof.
  intros Hin.
  apply ring_readable_mem_processed.
  - exact Hin.
  - exact (Nat.le_refl n).
Qed.

(* Node pool after nodes 0 through k-1 have executed their final
   loads. Memory remains unchanged throughout this sweep. *)
Fixpoint ring_final_np (n k : nat) : node_pool :=
  match k with
  | 0 => ring_second_np n n
  | S k' =>
      <[k' :=
        ([], <["x"%string := 1%Z]> (∅ : env))]>
        (ring_final_np n k')
  end.

Lemma ring_final_np_0 n :
  ring_final_np n 0 = ring_second_np n n.
Proof.
  reflexivity.
Qed.

Lemma ring_final_np_S n k :
  ring_final_np n (S k) =
    <[k :=
      ([], <["x"%string := 1%Z]> (∅ : env))]>
      (ring_final_np n k).
Proof.
  reflexivity.
Qed.

(* A node not yet visited by the load sweep still has its singleton
   final load and its empty environment. *)
Lemma ring_final_np_unprocessed n k i :
  k <= i ->
  i < n ->
  ring_final_np n k !! i =
    Some ([Load 1 "x"%string i 1], ∅).
Proof.
  intros Hki Hin.
  induction k as [|k IH].
  - rewrite ring_final_np_0.
    apply ring_second_np_processed.
    + exact Hin.
    + exact (Nat.le_refl n).
  - rewrite ring_final_np_S.
    transitivity (ring_final_np n k !! i).
    + apply lookup_insert_ne.
      lia.
    + apply IH.
      lia.
Qed.

(* A node already visited by the load sweep has no remaining
   instructions and has bound x to the loaded value 1. *)
Lemma ring_final_np_processed n k i :
  i < k ->
  k <= n ->
  ring_final_np n k !! i =
    Some ([], <["x"%string := 1%Z]> (∅ : env)).
Proof.
  intros Hik Hkn.
  induction k as [|k IH].
  - lia.
  - rewrite ring_final_np_S.
    destruct (Nat.eq_dec i k) as [-> | Hne].
    + apply lookup_insert_eq.
    + transitivity (ring_final_np n k !! i).
      * apply lookup_insert_ne.
        lia.
      * apply IH; lia.
Qed.

Lemma ring_final_np_lookup_in_bounds n k i p :
  k <= n ->
  ring_final_np n k !! i = Some p ->
  i < n.
Proof.
  intros Hkn.
  induction k as [|k IH].
  - rewrite ring_final_np_0.
    apply ring_second_np_lookup_in_bounds.
    exact (Nat.le_refl n).
  - intros Hlookup.
    rewrite ring_final_np_S in Hlookup.
    destruct (Nat.eq_dec k i) as [-> | Hne].
    + lia.
    + apply IH.
      * lia.
      * transitivity
          ((<[k :=
              ([], <["x"%string := 1%Z]> (∅ : env))]>
             (ring_final_np n k)) !! i).
        -- symmetry.
           apply lookup_insert_ne.
           exact Hne.
        -- exact Hlookup.
Qed.

Lemma ring_final_step n k :
  k < n ->
  step id (ring_graph n)
    (ring_final_np n k, ring_readable_mem n n)
    (ring_final_np n (S k), ring_readable_mem n n).
Proof.
  intros Hkn.
  pose proof
    (ring_final_np_unprocessed n k k (Nat.le_refl k) Hkn)
    as Hlookup.
  rewrite ring_final_np_S.
  eapply step_load.
  - exact Hlookup.
  - apply ring_load_allowed.
  - apply ring_readable_mem_final.
    exact Hkn.
Qed.

Lemma ring_final_sweep n k :
  k <= n ->
  rtc (step id (ring_graph n))
    (ring_second_np n n, ring_readable_mem n n)
    (ring_final_np n k, ring_readable_mem n n).
Proof.
  intros Hkn.
  induction k as [|k IH].
  - apply rtc_refl.
  - eapply rtc_r.
    + apply IH.
      lia.
    + apply ring_final_step.
      lia.
Qed.

(* All four transition sweeps compose. *)
Lemma ring_all_sweeps n :
  rtc (step id (ring_graph n))
    (init_config (ring n))
    (ring_final_np n n, ring_readable_mem n n).
Proof.
  etransitivity.
  - apply ring_store_readable_sweeps.
  - apply ring_final_sweep.
    exact (Nat.le_refl n).
Qed.

Lemma ring_final_finished n :
  finished (ring_final_np n n, ring_readable_mem n n).
Proof.
  unfold finished.
  cbn [fst].
  apply map_Forall_lookup_2.
  intros i p Hlookup.
  destruct p as [instructions e].
  assert (Hin : i < n).
  {
    eapply ring_final_np_lookup_in_bounds.
    - exact (Nat.le_refl n).
    - exact Hlookup.
  }
  pose proof
    (ring_final_np_processed n n i Hin (Nat.le_refl n))
    as Hexpected.
  pose proof (eq_trans (eq_sym Hlookup) Hexpected) as Heq.
  inversion Heq.
  reflexivity.
Qed.

Lemma ring_can_terminate n :
  exists c,
    rtc (step id (ring_graph n))
      (init_config (ring n)) c /\
    finished c.
Proof.
  exists (ring_final_np n n, ring_readable_mem n n).
  split.
  - apply ring_all_sweeps.
  - apply ring_final_finished.
Qed.

(* maybe we won't prove this *)
Lemma ring_can_always_terminate n : forall c, rtc (step id (ring_graph n)) (init_config (ring n)) c ->
  exists c', rtc (step id (ring_graph n)) c c' /\ finished c'.
Proof.
Abort.

Lemma ring_correct n : forall c, rtc (step id (ring_graph n)) (init_config (ring n)) c -> finished c ->
  map_Forall (λ _ '(_, e), e !! "x"%string = Some 1%Z) (fst c).
Proof.
  (* define an invariant, prove that it's preserved inductively *)
  (* Invariant: for each node i, either:
       it has all its instructions left, env is empty, location i+1@1 is 0
       it has 2 instructions left, env is empty, location i+1@1 is 1
       it has 1 instruction left, env is empty, location i+1@2 is 2
       it has no instructions left, env is "x" ↦ 1 *)
Abort.

Section status_semantics.

Variable owner : loc -> node_id.
Variable G : comm_graph.

(* This heartbeat model abstracts the runtime under the assumptions that issue times are
    monotonic and heartbeat and operation delivery is atomic.  Readability uses the
    directed minimum bound.  Discardability uses the directed maximum bound, but permits
    heartbeat evidence only for remote readers; local arbitrary look-back requires an
    explicit end-of-reads certificate.
   The single global timestamp map also abstracts the runtime's receiver- and region-local
   knowledge.  Removing finished nodes from readability checks stands for barrier completion;
   the runtime continues checking their clocks until its distributed barrier permits bypass. *)
Definition timestamp_of i :=
  match i with Load t _ _ _ | Store t _ _ _ => t end.

Definition live_nodes (np : node_pool) := dom (filter (λ '(_, (li, _)), li ≠ []) np).
Definition active_node (np : node_pool) (i : node_id) : Prop :=
  exists li e, np !! i = Some (li, e) /\ li <> [].
Definition config' := (node_pool * mem * gmap node_id timestamp)%type.

Definition node_pending_read
    (np : node_pool) (i : node_id) (l : loc) (t : timestamp) : Prop :=
  exists instructions e op,
    np !! i = Some (instructions, e) /\
    op ∈ instructions /\
    reads_from l t op.

Definition end_of_reads
    (np : node_pool) (i : node_id) (l : loc) (t : timestamp) : Prop :=
  ~ node_pending_read np i l t.

Definition readable_safe
    (np : node_pool) (nt : gmap node_id timestamp)
    (l : loc) (t : timestamp) : Prop :=
  forall i,
    active_node np i ->
    i = owner l \/ neighbors G i (owner l) ->
    exists ti, nt !! i = Some ti /\
      if Nat.eqb i (owner l)
      then t < ti
      else exists d,
        readable_lag G i (owner l) = Some d /\
        t < ti + d.

Definition discard_safe
    (np : node_pool) (nt : gmap node_id timestamp)
    (l : loc) (t : timestamp) : Prop :=
  forall i instructions e,
    np !! i = Some (instructions, e) ->
    i = owner l \/ neighbors G i (owner l) ->
    end_of_reads np i l t \/
    (i <> owner l /\
      exists ti d,
        nt !! i = Some ti /\
        discard_lag G i (owner l) = Some d /\
        t + d < ti).

Inductive step' : config' -> config' -> Prop :=
| step'_heartbeat np i op rest e m nt
  (Hi : np !! i = Some (op :: rest, e)):
    step' (np, m, nt)
          (np, m, <[i := timestamp_of op]>nt)
| step'_load np i t1 x l t rest e m v nt (Hi: np !! i = Some (Load t1 x l t :: rest, e))
  (Hcheck : operation_allowed owner i (Load t1 x l t) G)
  (Hload: m !! (l, t) = Some (v, Readable)):
    step' (np, m, nt) (<[i := (rest, <[x := v]>e)]>np, m, <[i := t1]>nt)
| step'_store np i t1 l t v rest e m nt
  (Hi : np !! i = Some (Store t1 l t v :: rest, e))
  (Hcheck : operation_allowed owner i (Store t1 l t v) G)
  (Hslot : m !! (l, t) = None \/
    exists v0, m !! (l, t) = Some (v0, Writeable)):
    step' (np, m, nt) (<[i := (rest, e)]>np, <[(l, t) := (v, Writeable)]>m, <[i := t1]>nt)
| step'_readable np m l t v nt
  (Hload : m !! (l, t) = Some (v, Writeable))
  (Hpast : readable_safe np nt l t):
  step' (np, m, nt) (np, <[(l, t) := (v, Readable)]>m, nt)
| step'_discardable np m l t v nt
  (Hload : m !! (l, t) = Some (v, Readable))
  (Hsafe : discard_safe np nt l t):
  step' (np, m, nt) (np, <[(l, t) := (v, Discardable)]>m, nt)
(* As written, this is still optional -- we can prove that it doesn't happen any earlier, but not that it
   happens as soon as it's ready. *)
.

Definition erase_status (c : config') : config :=
  let '(np, m, _) := c in (np, m).

Definition monotonic (li : list instr) := forall a b ia ib, a < b -> li !! a = Some ia -> li !! b = Some ib ->
  timestamp_of ia <= timestamp_of ib.

Definition monotonic_program (prog : gmap node_id (list instr)) : Prop :=
  map_Forall (fun _ li => monotonic li) prog.

Definition monotonic_node_pool (np : node_pool) : Prop :=
  map_Forall (fun _ p => monotonic (fst p)) np.

Definition allowed_node_pool (np : node_pool) : Prop :=
  map_Forall
    (fun i p => Forall (fun op => operation_allowed owner i op G) (fst p))
    np.

Definition allowed_program (prog : gmap node_id (list instr)) : Prop :=
  map_Forall
    (fun i li => Forall (fun op => operation_allowed owner i op G) li)
    prog.

Definition instruction_property_node_pool
    (P : node_id -> instr -> Prop) (np : node_pool) : Prop :=
  map_Forall (fun i p => Forall (P i) (fst p)) np.

Lemma instruction_property_init P prog :
  map_Forall (fun i li => Forall (P i) li) prog ->
  instruction_property_node_pool P (fst (init_config prog)).
Proof.
  intros Hproperty i [li e] Hlookup.
  unfold init_config in Hlookup.
  cbn [fst] in Hlookup.
  apply (proj1 (lookup_fmap_Some
    (fun ops : list instr => (ops, (∅ : env)))
    prog i (li, e))) in Hlookup.
  destruct Hlookup as [ops [Heq Hprog]].
  inversion Heq; subst.
  exact (Hproperty i li Hprog).
Qed.

Lemma instruction_property_abstract_step P np m np' m' :
  instruction_property_node_pool P np ->
  step owner G (np, m) (np', m') ->
  instruction_property_node_pool P np'.
Proof.
  intros Hproperty Hstep.
  inversion Hstep; subst.
  - apply map_Forall_insert_2.
    + cbn.
      apply Forall_inv_tail with (a := Load t1 x l t).
      exact (Hproperty i (Load t1 x l t :: rest, e) Hi).
    + exact Hproperty.
  - apply map_Forall_insert_2.
    + cbn.
      apply Forall_inv_tail with (a := Store t1 l t v).
      exact (Hproperty i (Store t1 l t v :: rest, e) Hi).
    + exact Hproperty.
  - exact Hproperty.
  - exact Hproperty.
Qed.

Lemma instruction_property_abstract_reachable P prog np m :
  map_Forall (fun i li => Forall (P i) li) prog ->
  rtc (step owner G) (init_config prog) (np, m) ->
  instruction_property_node_pool P np.
Proof.
  intros Hproperty Hrtc.
  pose proof (@rtc_ind_r config (step owner G)
    (fun c => instruction_property_node_pool P (fst c))
    (init_config prog)) as Hind.
  specialize (Hind (instruction_property_init P prog Hproperty)).
  specialize (Hind ltac:(
    intros [np1 m1] [np2 m2] _ Hstep IH;
    eapply instruction_property_abstract_step; eauto)).
  exact (Hind (np, m) Hrtc).
Qed.

Definition heartbeat_frontier (t : timestamp) (li : list instr) :=
  forall k op, li !! k = Some op -> t <= timestamp_of op.

Lemma monotonic_heartbeat_frontier op rest :
  monotonic (op :: rest) ->
  heartbeat_frontier (timestamp_of op) (op :: rest).
Proof.
  intros Hmonotonic k op' Hlookup.
  destruct k as [| k].
  - cbn in Hlookup.
    injection Hlookup as ->.
    lia.
  - eapply (Hmonotonic 0 (S k) op op').
    + lia.
    + reflexivity.
    + exact Hlookup.
Qed.

Lemma monotonic_tail op rest :
  monotonic (op :: rest) ->
  monotonic rest.
Proof.
  intros Hmonotonic a b ia ib Hab Hia Hib.
  eapply (Hmonotonic (S a) (S b) ia ib).
  - lia.
  - exact Hia.
  - exact Hib.
Qed.

Lemma monotonic_node_pool_init prog :
  monotonic_program prog ->
  monotonic_node_pool (fst (init_config prog)).
Proof.
  intros Hmono i [li e] Hlookup.
  unfold init_config in Hlookup.
  cbn [fst] in Hlookup.
  apply (proj1 (lookup_fmap_Some
    (fun ops : list instr => (ops, (∅ : env)))
    prog i (li, e))) in Hlookup.
  destruct Hlookup as [ops [Heq Hprog]].
  inversion Heq; subst.
  exact (Hmono i li Hprog).
Qed.

Lemma monotonic_node_pool_step np m np' m' :
  monotonic_node_pool np ->
  step owner G (np, m) (np', m') ->
  monotonic_node_pool np'.
Proof.
  intros Hmono Hstep.
  inversion Hstep; subst.
  - apply map_Forall_insert_2.
    + cbn. eapply monotonic_tail.
      exact (Hmono i (Load t1 x l t :: rest, e) Hi).
    + exact Hmono.
  - apply map_Forall_insert_2.
    + cbn. eapply monotonic_tail.
      exact (Hmono i (Store t1 l t v :: rest, e) Hi).
    + exact Hmono.
  - exact Hmono.
  - exact Hmono.
Qed.

Lemma monotonic_program_abstract_reachable prog np m :
  monotonic_program prog ->
  rtc (step owner G) (init_config prog) (np, m) ->
  monotonic_node_pool np.
Proof.
  intros Hmono Hrtc.
  pose proof (@rtc_ind_r config (step owner G)
    (fun c => monotonic_node_pool (fst c))
    (init_config prog)) as Hind.
  specialize (Hind (monotonic_node_pool_init prog Hmono)).
  specialize (Hind ltac:(
    intros [np1 m1] [np2 m2] _ Hstep IH;
    eapply monotonic_node_pool_step; eauto)).
  exact (Hind (np, m) Hrtc).
Qed.

Definition heartbeat_frontiers
    (np : node_pool) (nt : gmap node_id timestamp) : Prop :=
  forall i li e ti,
    np !! i = Some (li, e) ->
    nt !! i = Some ti ->
    heartbeat_frontier ti li.

Definition monotonic_status
    (np : node_pool) (nt : gmap node_id timestamp) : Prop :=
  map_Forall (fun _ p => monotonic (fst p)) np /\
  heartbeat_frontiers np nt.

Lemma status_discardability_no_pending_read np nt l t :
  monotonic_status np nt ->
  allowed_node_pool np ->
  discard_safe np nt l t ->
  ~ pending_read np l t.
Proof.
  intros [_ Hfront] Hallowed Hsafe Hpending.
  destruct Hpending as
    [i [instructions [e [op [Hlookup [Hop Hreads]]]]]].
  destruct op as [issued_at x l' target_at | issued_at l' target_at v].
  - cbn in Hreads.
    destruct Hreads as [-> ->].
    assert (Hop_allowed :
      operation_allowed owner i (Load issued_at x l t) G).
    {
      pose proof (Hallowed i (instructions, e) Hlookup) as Hall.
      cbn in Hall.
      rewrite Forall_forall in Hall.
      exact (Hall (Load issued_at x l t) Hop).
    }
    assert (Hrelevant : i = owner l \/ neighbors G i (owner l)).
    { exact (proj1 Hop_allowed). }
    destruct (Hsafe i instructions e Hlookup Hrelevant)
      as [Hended | [Hremote [ti [d [Hti [Hdiscard Hpast]]]]]].
    + apply Hended.
      exists instructions, e, (Load issued_at x l t).
      repeat split; assumption.
    + pose proof (Hfront i instructions e ti Hlookup Hti) as Hfront_i.
      apply elem_of_list_lookup_1 in Hop.
      destruct Hop as [k Hop].
      specialize (Hfront_i k (Load issued_at x l t) Hop).
      cbn in Hop_allowed.
      destruct Hop_allowed as [_ Htime].
      assert (Heqb : Nat.eqb (owner l) i = false).
      { apply Nat.eqb_neq. congruence. }
      rewrite Heqb in Htime.
      destruct Htime as
        [readable [discardable
          [Hreadable [Hdiscard' [Hlower Hupper]]]]].
      rewrite Hdiscard in Hdiscard'.
      injection Hdiscard' as <-.
      cbn in Hfront_i.
      lia.
  - exact Hreads.
Qed.

Lemma status_readability_visible np nt l t :
  monotonic_status np nt ->
  allowed_node_pool np ->
  readable_safe np nt l t ->
  visible np l t.
Proof.
  intros [_ Hfront] Hallowed Hpast Hpending.
  destruct Hpending as
    [i [instructions [e [op [Hlookup [Hop Hwrites]]]]]].
  destruct op as [issued_at x l' target_at | issued_at l' target_at v].
  - exact Hwrites.
  - cbn in Hwrites.
    destruct Hwrites as [-> ->].
    assert (Hactive : active_node np i).
    {
      exists instructions, e.
      split; first exact Hlookup.
      intros ->.
      rewrite elem_of_nil in Hop.
      contradiction.
    }
    assert (Hop_allowed :
      operation_allowed owner i (Store issued_at l t v) G).
    {
      pose proof (Hallowed i (instructions, e) Hlookup) as Hall.
      cbn in Hall.
      rewrite Forall_forall in Hall.
      exact (Hall (Store issued_at l t v) Hop).
    }
    assert (Hrelevant : i = owner l \/ neighbors G i (owner l)).
    {
      exact (proj1 Hop_allowed).
    }
    destruct (Hpast i Hactive Hrelevant) as [ti [Hti Hpast_i]].
    pose proof (Hfront i instructions e ti Hlookup Hti) as Hfront_i.
    apply elem_of_list_lookup_1 in Hop.
    destruct Hop as [k Hop].
    specialize (Hfront_i k (Store issued_at l t v) Hop).
    cbn in Hop_allowed.
    destruct Hop_allowed as [_ Htime].
    destruct (Nat.eq_dec (owner l) i) as [Hlocal | Hremote].
    + subst i.
      rewrite Nat.eqb_refl in Htime, Hpast_i.
      apply (Nat.lt_irrefl t).
      eapply Nat.lt_le_trans.
      * exact Hpast_i.
      * eapply Nat.le_trans.
        -- exact Hfront_i.
        -- exact Htime.
    + assert (Heqb : Nat.eqb (owner l) i = false).
      { apply Nat.eqb_neq. exact Hremote. }
      rewrite Heqb in Htime.
      assert (Heqb' : Nat.eqb i (owner l) = false).
      { apply Nat.eqb_neq. congruence. }
      rewrite Heqb' in Hpast_i.
      destruct Htime as [d [Hlag Hbound]].
      destruct Hpast_i as [d' [Hlag' Hpast']].
      rewrite Hlag in Hlag'.
      injection Hlag' as <-.
      apply (Nat.lt_irrefl t).
      eapply Nat.lt_le_trans.
      * exact Hpast'.
      * eapply Nat.le_trans.
        -- apply Nat.add_le_mono_r.
           exact Hfront_i.
        -- exact Hbound.
Qed.

Theorem abstract_step_lifts_with_evidence np m nt np' m' :
  step owner G (np, m) (np', m') ->
  (forall l t v,
    np' = np ->
    m' = <[(l, t) := (v, Readable)]>m ->
    readable_safe np nt l t) ->
  (forall l t v,
    np' = np ->
    m' = <[(l, t) := (v, Discardable)]>m ->
    discard_safe np nt l t) ->
  exists nt', step' (np, m, nt) (np', m', nt').
Proof.
  intros Hstep Hreadable Hdiscardable.
  inversion Hstep; subst.
  - exists (<[i := t1]>nt).
    eapply step'_load; eassumption.
  - exists (<[i := t1]>nt).
    eapply step'_store; eassumption.
  - exists nt.
    eapply step'_readable.
    + exact Hload.
    + eapply Hreadable; reflexivity.
  - exists nt.
    eapply step'_discardable.
    + exact Hload.
    + eapply Hdiscardable; reflexivity.
Qed.

Lemma status_step_refines_abstract np m nt np' m' nt' :
  monotonic_status np nt ->
  allowed_node_pool np ->
  step' (np, m, nt) (np', m', nt') ->
  erase_status (np, m, nt) = erase_status (np', m', nt') \/
  step owner G
    (erase_status (np, m, nt))
    (erase_status (np', m', nt')).
Proof.
  intros Hmono Hallowed Hstep.
  inversion Hstep; subst.
  - left. reflexivity.
  - right. cbn. eapply step_load; eassumption.
  - right. cbn. eapply step_store; eassumption.
  - right. cbn. eapply step_readable.
    + exact Hload.
    + eapply status_readability_visible; eassumption.
  - right. cbn. eapply step_discardable.
    + exact Hload.
    + eapply status_discardability_no_pending_read; eassumption.
Qed.

Lemma monotonic_status_initial np :
  map_Forall (fun _ p => monotonic (fst p)) np ->
  monotonic_status np empty.
Proof.
  intros Hmono.
  split; first exact Hmono.
  intros i li e ti Hnp Hnt.
  rewrite lookup_empty in Hnt.
  discriminate.
Qed.

Lemma init_config_monotonic prog :
  monotonic_program prog ->
  map_Forall
    (fun _ p => monotonic (fst p))
    (fst (init_config prog)).
Proof.
  intros Hmono i [li e] Hlookup.
  unfold init_config in Hlookup.
  cbn [fst] in Hlookup.
  apply (proj1 (lookup_fmap_Some
    (fun ops : list instr => (ops, (∅ : env)))
    prog i (li, e))) in Hlookup.
  destruct Hlookup as [ops [Heq Hprog]].
  inversion Heq; subst.
  exact (Hmono i li Hprog).
Qed.

Corollary init_config_monotonic_status prog :
  monotonic_program prog ->
  monotonic_status (fst (init_config prog)) empty.
Proof.
  intros Hmono.
  apply monotonic_status_initial.
  apply init_config_monotonic.
  exact Hmono.
Qed.

Lemma init_config_allowed prog :
  allowed_program prog ->
  allowed_node_pool (fst (init_config prog)).
Proof.
  intros Hallowed i [li e] Hlookup.
  unfold init_config in Hlookup.
  cbn [fst] in Hlookup.
  apply (proj1 (lookup_fmap_Some
    (fun ops : list instr => (ops, (∅ : env)))
    prog i (li, e))) in Hlookup.
  destruct Hlookup as [ops [Heq Hprog]].
  inversion Heq; subst.
  exact (Hallowed i li Hprog).
Qed.

Lemma allowed_program_reachable_head_store_steps
    prog np m i issued_at l target_at v rest e :
  allowed_program prog ->
  rtc (step owner G) (init_config prog) (np, m) ->
  np !! i = Some (Store issued_at l target_at v :: rest, e) ->
  step owner G (np, m)
    (<[i := (rest, e)]> np,
     <[(l, target_at) := (v, Writeable)]> m).
Proof.
  intros Hprogram Hrtc Hi.
  eapply step_store.
  - exact Hi.
  - pose proof
      (instruction_property_abstract_reachable
        (fun i op => operation_allowed owner i op G)
        prog np m Hprogram Hrtc)
      as Hallowed.
    pose proof
      (Hallowed i (Store issued_at l target_at v :: rest, e) Hi)
      as Hops.
    apply Forall_inv in Hops. exact Hops.
  - eapply reachable_head_store_target_available; eassumption.
Qed.

Theorem allowed_program_reachable_head_load_progress
    prog np m i issued_at x l target_at rest e :
  allowed_program prog ->
  load_versions_produced prog ->
  rtc (step owner G) (init_config prog) (np, m) ->
  np !! i = Some (Load issued_at x l target_at :: rest, e) ->
  visible np l target_at ->
  (exists v,
    m !! (l, target_at) = Some (v, Readable) /\
    step owner G (np, m)
      (<[i := (rest, <[x := v]> e)]> np, m)) \/
  (exists v,
    m !! (l, target_at) = Some (v, Writeable) /\
    step owner G (np, m)
      (np, <[(l, target_at) := (v, Readable)]> m) /\
    step owner G
      (np, <[(l, target_at) := (v, Readable)]> m)
      (<[i := (rest, <[x := v]> e)]> np,
       <[(l, target_at) := (v, Readable)]> m)).
Proof.
  intros Hprogram Hproduced Hrtc Hi Hvisible.
  assert (Hpending : pending_read np l target_at).
  {
    exists i, (Load issued_at x l target_at :: rest), e,
      (Load issued_at x l target_at).
    repeat split.
    - exact Hi.
    - apply elem_of_cons. left. reflexivity.
  }
  pose proof
    (pending_loads_available_reachable
      owner G prog np m Hproduced Hrtc l target_at Hpending)
    as Havailable.
  destruct Havailable as [Hwrite | [v [Hwriteable | Hreadable]]].
  - exact (False_rect _ (Hvisible Hwrite)).
  - right. exists v. split; first exact Hwriteable.
    split.
    + eapply step_readable; eassumption.
    + eapply step_load.
      * exact Hi.
      * pose proof
          (instruction_property_abstract_reachable
            (fun i op => operation_allowed owner i op G)
            prog np m Hprogram Hrtc)
          as Hallowed.
        pose proof
          (Hallowed i (Load issued_at x l target_at :: rest, e) Hi)
          as Hops.
        apply Forall_inv in Hops. exact Hops.
      * apply lookup_insert.
  - left. exists v. split; first exact Hreadable.
    eapply step_load.
    + exact Hi.
    + pose proof
        (instruction_property_abstract_reachable
          (fun i op => operation_allowed owner i op G)
          prog np m Hprogram Hrtc)
        as Hallowed.
      pose proof
        (Hallowed i (Load issued_at x l target_at :: rest, e) Hi)
        as Hops.
      apply Forall_inv in Hops. exact Hops.
    + exact Hreadable.
Qed.

Lemma allowed_node_pool_step np m nt np' m' nt' :
  allowed_node_pool np ->
  step' (np, m, nt) (np', m', nt') ->
  allowed_node_pool np'.
Proof.
  intros Hallowed Hstep.
  inversion Hstep; subst.
  - exact Hallowed.
  - apply map_Forall_insert_2.
    + cbn.
      apply Forall_inv_tail with (a := Load t1 x l t).
      exact (Hallowed i (Load t1 x l t :: rest, e) Hi).
    + exact Hallowed.
  - apply map_Forall_insert_2.
    + cbn.
      apply Forall_inv_tail with (a := Store t1 l t v).
      exact (Hallowed i (Store t1 l t v :: rest, e) Hi).
    + exact Hallowed.
  - exact Hallowed.
  - exact Hallowed.
Qed.

Lemma allowed_node_pool_reachable np m nt np' m' nt' :
  allowed_node_pool np ->
  rtc step' (np, m, nt) (np', m', nt') ->
  allowed_node_pool np'.
Proof.
  intros Hallowed Hrtc.
  pose proof (@rtc_ind_r config' step'
    (fun c => allowed_node_pool c.1.1) (np, m, nt)) as Hind.
  specialize (Hind Hallowed).
  specialize (Hind ltac:(
    intros [[np1 m1] nt1] [[np2 m2] nt2] _ Hstep IH;
    eapply allowed_node_pool_step; eauto)).
  exact (Hind (np', m', nt') Hrtc).
Qed.

Corollary allowed_program_status_reachable prog np' m' nt' :
  allowed_program prog ->
  rtc step'
    (fst (init_config prog), snd (init_config prog), empty)
    (np', m', nt') ->
  allowed_node_pool np'.
Proof.
  intros Hallowed Hrtc.
  eapply allowed_node_pool_reachable.
  - apply init_config_allowed.
    exact Hallowed.
  - exact Hrtc.
Qed.

Lemma monotonic_status_step np m nt np' m' nt' :
  monotonic_status np nt ->
  step' (np, m, nt) (np', m', nt') ->
  monotonic_status np' nt'.
Proof.
  intros [Hmono Hfront] Hstep.
  inversion Hstep; subst.
  - split; first exact Hmono.
    intros j li e' tj Hnp Hnt.
    destruct (decide (j = i)) as [-> | Hne].
    + rewrite lookup_insert in Hnt.
      injection Hnt as <-.
      rewrite Hi in Hnp.
      injection Hnp as <- <-.
      apply monotonic_heartbeat_frontier.
      exact (Hmono i (op :: rest, e) Hi).
    + rewrite lookup_insert_ne in Hnt; last congruence.
      eapply Hfront; eauto.
  - split.
    + apply map_Forall_insert_2.
      * apply monotonic_tail with (op := Load t1 x l t).
        exact (Hmono i (Load t1 x l t :: rest, e) Hi).
      * exact Hmono.
    + intros j li e' tj Hnp Hnt.
      destruct (decide (j = i)) as [-> | Hne].
      * rewrite lookup_insert in Hnt.
        injection Hnt as <-.
        assert (<[i := (rest, <[x := v]> e)]> np !! i =
          Some (rest, <[x := v]> e)) as Hins by apply lookup_insert.
        rewrite Hnp in Hins.
        inversion Hins; subst.
        intros k op' Hlookup.
        pose proof
          (monotonic_heartbeat_frontier (Load t1 x l t) rest) as Hf.
        specialize (Hf (Hmono i (Load t1 x l t :: rest, e) Hi)).
        exact (Hf (S k) op' Hlookup).
      * rewrite lookup_insert_ne in Hnt; last congruence.
        assert (np !! j = Some (li, e')) as Hnp_old.
        { rewrite <- Hnp. symmetry. apply lookup_insert_ne. congruence. }
        eapply Hfront; eauto.
  - split.
    + apply map_Forall_insert_2.
      * apply monotonic_tail with (op := Store t1 l t v).
        exact (Hmono i (Store t1 l t v :: rest, e) Hi).
      * exact Hmono.
    + intros j li e' tj Hnp Hnt.
      destruct (decide (j = i)) as [-> | Hne].
      * rewrite lookup_insert in Hnt.
        injection Hnt as <-.
        assert (<[i := (rest, e)]> np !! i = Some (rest, e)) as Hins
          by apply lookup_insert.
        rewrite Hnp in Hins.
        inversion Hins; subst.
        intros k op' Hlookup.
        pose proof
          (monotonic_heartbeat_frontier (Store t1 l t v) rest) as Hf.
        specialize (Hf (Hmono i (Store t1 l t v :: rest, e) Hi)).
        exact (Hf (S k) op' Hlookup).
      * rewrite lookup_insert_ne in Hnt; last congruence.
        assert (np !! j = Some (li, e')) as Hnp_old.
        { rewrite <- Hnp. symmetry. apply lookup_insert_ne. congruence. }
        eapply Hfront; eauto.
  - split; assumption.
  - split; assumption.
Qed.

Lemma status_reachable_refines_abstract np m nt np' m' nt' :
  monotonic_status np nt ->
  allowed_node_pool np ->
  rtc step' (np, m, nt) (np', m', nt') ->
  rtc (step owner G) (np, m) (np', m').
Proof.
  intros Hmono Hallowed Hrtc.
  pose proof (@rtc_ind_r config' step'
    (fun c =>
      rtc (step owner G) (np, m) (erase_status c) /\
      monotonic_status c.1.1 c.2 /\
      allowed_node_pool c.1.1)
    (np, m, nt)) as Hind.
  specialize (Hind ltac:(
    split;
    [ apply rtc_refl
    | split; assumption ])).
  specialize (Hind ltac:(
    intros [[np1 m1] nt1] [[np2 m2] nt2] _ Hstep
      [Habstract [Hmono1 Hallowed1]];
    split;
    [ destruct (status_step_refines_abstract
        np1 m1 nt1 np2 m2 nt2 Hmono1 Hallowed1 Hstep)
        as [Herase | Hmodel];
      [ rewrite <- Herase; exact Habstract
      | eapply rtc_r; eassumption ]
    | split;
      [ eapply monotonic_status_step; eassumption
      | eapply allowed_node_pool_step; eassumption ] ])).
  exact (proj1 (Hind (np', m', nt') Hrtc)).
Qed.

Lemma monotonic_status_reachable np m np' m' nt' :
  map_Forall (fun _ p => monotonic (fst p)) np ->
  rtc step' (np, m, empty) (np', m', nt') ->
  monotonic_status np' nt'.
Proof.
  intros Hmono Hrtc.
  pose proof (@rtc_ind_r config' step'
    (fun c => monotonic_status c.1.1 c.2) (np, m, empty)) as Hind.
  specialize (Hind (monotonic_status_initial np Hmono)).
  specialize (Hind ltac:(
    intros [[np1 m1] nt1] [[np2 m2] nt2] _ Hstep IH;
    eapply monotonic_status_step; eauto)).
  exact (Hind (np', m', nt') Hrtc).
Qed.

Corollary reachable_heartbeat_frontiers np m np' m' nt' :
  map_Forall (fun _ p => monotonic (fst p)) np ->
  rtc step' (np, m, empty) (np', m', nt') ->
  heartbeat_frontiers np' nt'.
Proof.
  intros Hmono Hrtc.
  exact (proj2 (monotonic_status_reachable
    np m np' m' nt' Hmono Hrtc)).
Qed.

Corollary monotonic_program_status_reachable
    prog np' m' nt' :
  monotonic_program prog ->
  rtc step'
    (fst (init_config prog), snd (init_config prog), empty)
    (np', m', nt') ->
  monotonic_status np' nt'.
Proof.
  intros Hmono Hrtc.
  eapply monotonic_status_reachable.
  - apply init_config_monotonic.
    exact Hmono.
  - exact Hrtc.
Qed.

(* Intended completeness direction: if all nodes' programs are monotonic, then
   the status semantics can reproduce anything the nondeterministic semantics
   can do.  The current premises are insufficient: an abstract readability
   step may be enabled without a strictly later heartbeat or validated barrier
   evidence.  abstract_step_lifts_with_evidence proves the justified
   single-step form once that evidence is supplied. *)
Theorem monotonic_step'_safe np : map_Forall (λ _ '(li, _), monotonic li) np ->
  forall m np' m', rtc (step owner G) (np, m) (np', m') ->
  exists nt', rtc step' (np, m, empty) (np', m', nt').
Proof.
Abort.

End status_semantics.

(* further refinement: travel time between nodes, and its relationship to read/write limits *)

(* Properties:
1. A valid program admits a schedule under which it terminates.
2. Anything that can be expressed with conservative synchronization can be expressed in DSTM.
   (only one live timestamp at a time; write syntax and semantics, a translation function, and prove
    that the translation preserves semantics)
3. Programs using the push model/pull model won't deadlock.
4. The status semantics correctly implements the more abstract semantics (which makes locations Readable
   precisely when it needs to).
*)

(* Here, valid means push (remote write at t+d, local read at t) or pull (local write at t,
   remote read at t-d), using the directed channel lag d.  Deadlock freedom means every reachable state is finished or can step.
   Multiple writes may overwrite a Writeable version.  The implementation broadcasts advancing
   logical times as heartbeats and uses per-region neighbor times plus distance to close versions. *)

Definition push_instr
    (owner : loc -> node_id) (G : comm_graph)
    (issuer : node_id) (op : instr) : Prop :=
  operation_allowed owner issuer op G /\
  match op with
  | Load issued_at _ l target_at =>
      owner l = issuer /\ target_at = issued_at
  | Store issued_at l target_at _ =>
      owner l <> issuer /\
      exists d,
        readable_lag G issuer (owner l) = Some d /\
        target_at = issued_at + d
  end.

Definition pull_instr
    (owner : loc -> node_id) (G : comm_graph)
    (issuer : node_id) (op : instr) : Prop :=
  operation_allowed owner issuer op G /\
  match op with
  | Load issued_at _ l target_at =>
      owner l <> issuer /\
      exists d,
        readable_lag G issuer (owner l) = Some d /\
        issued_at = target_at + d
  | Store issued_at l target_at _ =>
      owner l = issuer /\ target_at = issued_at
  end.

Definition push_program
    (owner : loc -> node_id) (G : comm_graph)
    (prog : gmap node_id (list instr)) : Prop :=
  map_Forall (λ issuer ops, Forall (push_instr owner G issuer) ops) prog.

Definition pull_program
    (owner : loc -> node_id) (G : comm_graph)
    (prog : gmap node_id (list instr)) : Prop :=
  map_Forall (λ issuer ops, Forall (pull_instr owner G issuer) ops) prog.

Lemma push_load_store_dependency_decreases
    owner G reader writer load_at x l target_at store_at v :
  push_instr owner G reader (Load load_at x l target_at) ->
  push_instr owner G writer (Store store_at l target_at v) ->
  store_at < load_at.
Proof.
  intros Hload Hstore.
  destruct Hload as [_ [Hlocal ->]].
  destruct Hstore as [_ [_ [d [Hlag Htarget]]]].
  pose proof (channel_positive G writer (owner l) d Hlag).
  lia.
Qed.

Lemma pull_load_store_dependency_decreases
    owner G reader writer load_at x l target_at store_at v :
  pull_instr owner G reader (Load load_at x l target_at) ->
  pull_instr owner G writer (Store store_at l target_at v) ->
  store_at < load_at.
Proof.
  intros Hload Hstore.
  destruct Hload as [_ [_ [d [Hlag Hload_time]]]].
  destruct Hstore as [_ [_ Hstore_time]].
  pose proof (channel_positive G reader (owner l) d Hlag).
  lia.
Qed.

Lemma pending_write_has_no_later_head np l t :
  monotonic_node_pool np ->
  pending_write np l t ->
  exists writer head rest e store_at v,
    np !! writer = Some (head :: rest, e) /\
    Store store_at l t v ∈ head :: rest /\
    timestamp_of head <= store_at.
Proof.
  intros Hmono Hpending.
  destruct Hpending as
    [writer [instructions [e [op [Hlookup [Hop Hwrites]]]]]].
  destruct op as [load_at x l' target_at | store_at l' target_at v].
  - exact (False_rect _ Hwrites).
  - cbn in Hwrites. destruct Hwrites as [-> ->].
    destruct instructions as [|head rest].
    + rewrite elem_of_nil in Hop. contradiction.
    + exists writer, head, rest, e, store_at, v.
      repeat split; try assumption.
      pose proof (Hmono writer (head :: rest, e) Hlookup) as Hlist.
      pose proof (monotonic_heartbeat_frontier head rest Hlist) as Hfront.
      apply elem_of_list_lookup_1 in Hop.
      destruct Hop as [k Hop].
      exact (Hfront k (Store store_at l t v) Hop).
Qed.

Theorem push_program_head_load_pending_write_descends
    owner G prog np m reader load_at x l target_at load_rest load_env :
  monotonic_program prog ->
  push_program owner G prog ->
  rtc (@step owner G) (init_config prog) (np, m) ->
  np !! reader = Some (Load load_at x l target_at :: load_rest, load_env) ->
  pending_write np l target_at ->
  exists writer head rest e,
    np !! writer = Some (head :: rest, e) /\
    timestamp_of head < load_at.
Proof.
  intros Hmono Hpush Hrtc Hload Hpending.
  pose proof (monotonic_program_abstract_reachable
    owner G prog np m Hmono Hrtc) as Hmono_np.
  destruct (pending_write_has_no_later_head
    np l target_at Hmono_np Hpending)
    as [writer [head [rest [e [store_at [v [Hwriter [Hstore Hhead]]]]]]]].
  pose proof (instruction_property_abstract_reachable
    owner G (push_instr owner G) prog np m Hpush Hrtc) as Hpush_np.
  pose proof (Hpush_np reader
    (Load load_at x l target_at :: load_rest, load_env) Hload) as Hloads.
  pose proof (Hpush_np writer (head :: rest, e) Hwriter) as Hstores.
  apply Forall_inv in Hloads.
  rewrite Forall_forall in Hstores.
  pose proof (Hstores (Store store_at l target_at v) Hstore) as Hstore_push.
  exists writer, head, rest, e.
  split; first exact Hwriter.
  pose proof (push_load_store_dependency_decreases
    owner G reader writer load_at x l target_at store_at v
    Hloads Hstore_push).
  lia.
Qed.

Theorem pull_program_head_load_pending_write_descends
    owner G prog np m reader load_at x l target_at load_rest load_env :
  monotonic_program prog ->
  pull_program owner G prog ->
  rtc (@step owner G) (init_config prog) (np, m) ->
  np !! reader = Some (Load load_at x l target_at :: load_rest, load_env) ->
  pending_write np l target_at ->
  exists writer head rest e,
    np !! writer = Some (head :: rest, e) /\
    timestamp_of head < load_at.
Proof.
  intros Hmono Hpull Hrtc Hload Hpending.
  pose proof (monotonic_program_abstract_reachable
    owner G prog np m Hmono Hrtc) as Hmono_np.
  destruct (pending_write_has_no_later_head
    np l target_at Hmono_np Hpending)
    as [writer [head [rest [e [store_at [v [Hwriter [Hstore Hhead]]]]]]]].
  pose proof (instruction_property_abstract_reachable
    owner G (pull_instr owner G) prog np m Hpull Hrtc) as Hpull_np.
  pose proof (Hpull_np reader
    (Load load_at x l target_at :: load_rest, load_env) Hload) as Hloads.
  pose proof (Hpull_np writer (head :: rest, e) Hwriter) as Hstores.
  apply Forall_inv in Hloads.
  rewrite Forall_forall in Hstores.
  pose proof (Hstores (Store store_at l target_at v) Hstore) as Hstore_pull.
  exists writer, head, rest, e.
  split; first exact Hwriter.
  pose proof (pull_load_store_dependency_decreases
    owner G reader writer load_at x l target_at store_at v
    Hloads Hstore_pull).
  lia.
Qed.

Definition valid_program
    (owner : loc -> node_id) (G : comm_graph)
    (prog : gmap node_id (list instr)) : Prop :=
  monotonic_program prog /\
  load_versions_produced prog /\
  (push_program owner G prog \/ pull_program owner G prog).

Lemma push_program_allowed owner G prog :
  push_program owner G prog -> allowed_program owner G prog.
Proof.
  intros Hpush i ops Hlookup.
  rewrite Forall_forall.
  intros op Hop.
  pose proof (Hpush i ops Hlookup) as Hops.
  rewrite Forall_forall in Hops.
  exact (proj1 (Hops op Hop)).
Qed.

Lemma pull_program_allowed owner G prog :
  pull_program owner G prog -> allowed_program owner G prog.
Proof.
  intros Hpull i ops Hlookup.
  rewrite Forall_forall.
  intros op Hop.
  pose proof (Hpull i ops Hlookup) as Hops.
  rewrite Forall_forall in Hops.
  exact (proj1 (Hops op Hop)).
Qed.

Lemma valid_program_allowed owner G prog :
  valid_program owner G prog -> allowed_program owner G prog.
Proof.
  intros [_ [_ [Hpush | Hpull]]].
  - exact (push_program_allowed owner G prog Hpush).
  - exact (pull_program_allowed owner G prog Hpull).
Qed.

Theorem push_program_reachable_head_store_progress
    owner G prog np m i issued_at l target_at v rest e :
  push_program owner G prog ->
  rtc (@step owner G) (init_config prog) (np, m) ->
  np !! i = Some (Store issued_at l target_at v :: rest, e) ->
  owner l <> i /\
  step owner G (np, m)
    (<[i := (rest, e)]> np,
     <[(l, target_at) := (v, Writeable)]> m).
Proof.
  intros Hpush Hrtc Hi.
  assert (Hallowed : allowed_program owner G prog).
  { exact (push_program_allowed owner G prog Hpush). }
  split.
  - pose proof
      (instruction_property_abstract_reachable
        owner G (push_instr owner G) prog np m Hpush Hrtc) as Hpool.
    pose proof
      (Hpool i (Store issued_at l target_at v :: rest, e) Hi)
      as Hops.
    apply Forall_inv in Hops.
    exact (proj1 (proj2 Hops)).
  - eapply allowed_program_reachable_head_store_steps; eassumption.
Qed.

Theorem pull_program_reachable_head_store_progress
    owner G prog np m i issued_at l target_at v rest e :
  pull_program owner G prog ->
  rtc (@step owner G) (init_config prog) (np, m) ->
  np !! i = Some (Store issued_at l target_at v :: rest, e) ->
  owner l = i /\
  step owner G (np, m)
    (<[i := (rest, e)]> np,
     <[(l, target_at) := (v, Writeable)]> m).
Proof.
  intros Hpull Hrtc Hi.
  assert (Hallowed : allowed_program owner G prog).
  { exact (pull_program_allowed owner G prog Hpull). }
  split.
  - pose proof
      (instruction_property_abstract_reachable
        owner G (pull_instr owner G) prog np m Hpull Hrtc) as Hpool.
    pose proof
      (Hpool i (Store issued_at l target_at v :: rest, e) Hi)
      as Hops.
    apply Forall_inv in Hops.
    exact (proj1 (proj2 Hops)).
  - eapply allowed_program_reachable_head_store_steps; eassumption.
Qed.

Corollary valid_program_status_reachable owner G prog np' m' nt' :
  valid_program owner G prog ->
  rtc (@step' owner G)
    (fst (init_config prog), snd (init_config prog), empty)
    (np', m', nt') ->
  allowed_node_pool owner G np'.
Proof.
  intros Hvalid Hrtc.
  eapply allowed_program_status_reachable.
  - exact (valid_program_allowed owner G prog Hvalid).
  - exact Hrtc.
Qed.

Theorem valid_status_reachable_abstract owner G prog np' m' nt' :
  valid_program owner G prog ->
  rtc (@step' owner G)
    (fst (init_config prog), snd (init_config prog), empty)
    (np', m', nt') ->
  rtc (@step owner G) (init_config prog) (np', m').
Proof.
  intros Hvalid Hrtc.
  eapply status_reachable_refines_abstract.
  - apply init_config_monotonic_status.
    exact (proj1 Hvalid).
  - apply init_config_allowed.
    exact (valid_program_allowed owner G prog Hvalid).
  - exact Hrtc.
Qed.

Corollary valid_program_status_invariants owner G prog np' m' nt' :
  valid_program owner G prog ->
  rtc (@step' owner G)
    (fst (init_config prog), snd (init_config prog), empty)
    (np', m', nt') ->
  monotonic_status np' nt' /\ allowed_node_pool owner G np'.
Proof.
  intros Hvalid Hrtc.
  split.
  - eapply monotonic_program_status_reachable.
    + exact (proj1 Hvalid).
    + exact Hrtc.
  - exact (valid_program_status_reachable
      owner G prog np' m' nt' Hvalid Hrtc).
Qed.

Corollary valid_status_discardability_no_pending_read
    owner G prog np' m' nt' l t :
  valid_program owner G prog ->
  rtc (@step' owner G)
    (fst (init_config prog), snd (init_config prog), empty)
    (np', m', nt') ->
  discard_safe owner G np' nt' l t ->
  ~ pending_read np' l t.
Proof.
  intros Hvalid Hrtc Hsafe.
  destruct (valid_program_status_invariants
    owner G prog np' m' nt' Hvalid Hrtc) as [Hmono Hallowed].
  eapply status_discardability_no_pending_read; eauto.
Qed.

Corollary valid_status_readability_visible
    owner G prog np' m' nt' l t :
  valid_program owner G prog ->
  rtc (@step' owner G)
    (fst (init_config prog), snd (init_config prog), empty)
    (np', m', nt') ->
  readable_safe owner G np' nt' l t ->
  visible np' l t.
Proof.
  intros Hvalid Hrtc Hpast.
  destruct (valid_program_status_invariants
    owner G prog np' m' nt' Hvalid Hrtc) as [Hmono Hallowed].
  eapply status_readability_visible; eauto.
Qed.
