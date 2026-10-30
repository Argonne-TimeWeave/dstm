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

(* A non-local successor in the ring is connected by an edge with lag 1. *)
Lemma ring_successor_lag n i :
  i < n ->
  (i + 1) `mod` n <> i ->
  channel_lag (ring_graph n) i ((i + 1) `mod` n) = Some 1.
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
         channel_lag (ring_graph n) i ((i + 1) `mod` n) = Some 1).
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

(* Properties:
1. A valid program admits a schedule under which it terminates.
2. Anything that can be expressed with conservative synchronization can be expressed in DSTM.
   (only one live timestamp at a time; write syntax and semantics, a translation function, and prove
    that the translation preserves semantics)
3. Programs using the push model/pull model won't deadlock.
4. The status semantics correctly implements the more abstract semantics (which makes locations Readable
   precisely when it needs to).
*)
