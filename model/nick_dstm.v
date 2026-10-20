Require Import String.
Require Import Lia.
Require Import stdpp.gmap.

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

  (* None corresponds to C values -1 or 0; Some d to d > 0. *)
  channel_lag : node_id -> node_id -> option nat;

  channel_in_bounds :
    forall p q d,
      channel_lag p q = Some d ->
      p < world_size /\ q < world_size;

  channel_irreflexive : 
    forall p, p < world_size -> 
      channel_lag p p = None;

  channel_positive : 
    forall p q d, channel_lag p q = Some d -> d > 0
}.

Definition symmetric_graph (G : comm_graph) : Prop := 
  forall p q,
    p < world_size G -> 
    q < world_size G ->
    channel_lag G p q = channel_lag G q p.

Definition neighbors (G : comm_graph) (p q : node_id) : Prop :=
  exists d, channel_lag G p q = Some d.

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
        exists d, 
          channel_lag G issuer (owner l) = Some d /\
          target_at + d <= issued_at
  | Store issued_at l target_at _ =>
      target_allowed G issuer (owner l) /\
      if Nat.eqb (owner l) issuer
      then 
        issued_at <= target_at
      else
        exists d,
          channel_lag G issuer (owner l) = Some d /\
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

Inductive mem_status := Writeable | Readable (*| Discardable*).
Definition mem := gmap (loc * timestamp) (val * mem_status). (* revisit this *)
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
(* It's the runtime's responsibility to ensure that this is done at the right time to make the program work. *)
.

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

(* Once a version is Readable, one execution step preserves both
   its value and its Readable status. *)
Lemma readable_preserved_step
    (np np' : node_pool) (m m' : mem)
    (l : loc) (t : timestamp) (v : val)
    (Hread : m !! (l, t) = Some (v, Readable))
    (Hstep : step (np, m) (np', m')) :
  m' !! (l, t) = Some (v, Readable).
Proof.
  inversion Hstep; subst.
  - (* Load: memory is unchanged. *)
    exact Hread.

  - (* Store. *)
    destruct (decide ((l0, t0) = (l, t))) as [Heq | Hneq].
    + inversion Heq; subst.
      destruct Hslot as [Habsent | [v_old Hwriteable]].
      * rewrite Hread in Habsent.
        discriminate.
      * rewrite Hread in Hwriteable.
        discriminate.
    + transitivity (m !! (l, t)).
      * apply lookup_insert_ne.
        congruence.
      * exact Hread.

  - (* Readability transition. *)
    destruct (decide ((l0, t0) = (l, t))) as [Heq | Hneq].
    + inversion Heq; subst.
      rewrite Hread in Hload.
      discriminate.
    + transitivity (m !! (l, t)).
      * apply lookup_insert_ne.
        congruence.
      * exact Hread.
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

(* A permitted remote load reads a version at least one channel lag
   behind its issuing timestamp. *)
Lemma remote_load_targets_old_version
    (issuer : node_id) (issued_at : timestamp)
    (x : var) (l : loc) (target_at : timestamp)
    (Hremote : issuer <> owner l)
    (Hallowed :
       operation_allowed owner issuer
         (Load issued_at x l target_at) G) :
  exists d,
    channel_lag G issuer (owner l) = Some d /\
    target_at + d <= issued_at /\
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
  destruct Htime as [d [Hlag Hbound]].
  exists d.
  repeat split; try assumption.

  pose proof (channel_positive G issuer (owner l) d Hlag) as Hpositive.
  lia.
Qed.

(* A permitted remote store writes a version at least one channel lag
   ahead of its issuing timestamp. *)
Lemma remote_store_targets_future_version
    (issuer : node_id) (issued_at : timestamp)
    (l : loc) (target_at : timestamp) (v : val)
    (Hremote : issuer <> owner l)
    (Hallowed :
       operation_allowed owner issuer
         (Store issued_at l target_at v) G) :
  exists d,
    channel_lag G issuer (owner l) = Some d /\
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

End semantics.

(* Examples 

Ring(n):
for some number of timestamps t:
  at t, store (n + 1) (t + 1) (t + 1)
  at t, load n t , make sure the value is t

*)
Definition ring (n : nat) : gmap node_id (list instr) :=
  foldr (λ i prog, <[i := [Store 0 ((i + 1) `mod` n) 1 1%Z; (*Load 0 "x"%string i 0;*)
                           Store 1 ((i + 1) `mod` n) 2 2%Z; Load 1 "x"%string i 1]]>prog) ∅ (seq 0 n).

Definition neighbor n n1 n2 : bool := (n2 =? (n1 + 1) `mod` n) || (n2 =? (n1 + n - 1) `mod` n).

Program Definition ring_graph n : comm_graph := {|
  world_size := n;
  channel_lag n1 n2 := 
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

Lemma ring_can_terminate n : exists c, rtc (step id (ring_graph n)) (init_config (ring n)) c /\ finished c.
Proof.
  (* define a three-part schedule (running each instruction); prove each by induction on n *)
Abort.

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

Definition timestamp_of i :=
  match i with Load t _ _ _ | Store t _ _ _ => t end.

Definition live_nodes (np : node_pool) := dom (filter (λ '(_, (li, _)), li ≠ []) np).
Definition config' := (node_pool * mem * gmap node_id timestamp)%type.

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
  (Hpast : forall i,
    i ∈ live_nodes np ->
    i = owner l \/ neighbors G i (owner l) ->
    exists ti, nt !! i = Some ti /\
      if Nat.eqb i (owner l)
      then t <= ti
      else exists d,
        channel_lag G i (owner l) = Some d /\
        t < ti + d):
  step' (np, m, nt) (np, <[(l, t) := (v, Readable)]>m, nt)
(* As written, this is still optional -- we can prove that it doesn't happen any earlier, but not that it
   happens as soon as it's ready. *)
.

Definition monotonic (li : list instr) := forall a b ia ib, a < b -> li !! a = Some ia -> li !! b = Some ib ->
  timestamp_of ia <= timestamp_of ib.

(* If all nodes' programs are monotonic, then this implementation can do anything that the nondeterministic
   implementation can do. *)
Theorem monotonic_step'_safe np : map_Forall (λ _ '(li, _), monotonic li) np ->
  forall m np' m', rtc (step owner G) (np, m) (np', m') ->
  exists nt', rtc step' (np, m, empty) (np', m', nt').
Proof.
Abort.

End status_semantics.

(* further refinement: travel time between nodes, and its relationship to read/write limits *)
