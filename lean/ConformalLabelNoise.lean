namespace PaperA


def goodL : List (Bool × Bool) → Nat
  | [] => 0
  | (a, _) :: ps => (if a then 1 else 0) + goodL ps

def goodR : List (Bool × Bool) → Nat
  | [] => 0
  | (_, b) :: ps => (if b then 1 else 0) + goodR ps

def diffP : List (Bool × Bool) → Nat
  | [] => 0
  | (a, b) :: ps => (if a = b then 0 else 1) + diffP ps

theorem goodL_le_goodR_add_diff : ∀ ps : List (Bool × Bool),
    goodL ps ≤ goodR ps + diffP ps := by
  intro ps
  induction ps with
  | nil => simp [goodL, goodR, diffP]
  | cons p ps ih =>
    rcases p with ⟨a, b⟩
    cases a <;> cases b <;> simp [goodL, goodR, diffP] at ih ⊢ <;> omega

theorem goodR_le_goodL_add_diff : ∀ ps : List (Bool × Bool),
    goodR ps ≤ goodL ps + diffP ps := by
  intro ps
  induction ps with
  | nil => simp [goodL, goodR, diffP]
  | cons p ps ih =>
    rcases p with ⟨a, b⟩
    cases a <;> cases b <;> simp [goodL, goodR, diffP] at ih ⊢ <;> omega


def SortedScore : List Int → Prop
  | [] => True
  | a :: rest => (∀ b ∈ rest, a < b) ∧ SortedScore rest


def countLE (q : Int) : List Int → Nat
  | [] => 0
  | a :: as => (if a ≤ q then 1 else 0) + countLE q as


def orderStat : List Int → Nat → Int
  | [], _ => 0
  | _ :: _, 0 => 0
  | a :: _, 1 => a
  | _ :: as, Nat.succ (Nat.succ k) => orderStat as (Nat.succ k)

theorem countLE_le_length (q : Int) : ∀ s : List Int, countLE q s ≤ s.length := by
  intro s
  induction s with
  | nil => simp [countLE]
  | cons a as ih =>
    by_cases ha : a ≤ q
    · simp [countLE, ha]; omega
    · simp [countLE, ha]; omega

theorem countLE_head_zero {a : Int} : ∀ {as : List Int},
    (∀ b ∈ as, a < b) → countLE a as = 0 := by
  intro as
  induction as with
  | nil => intro _; rfl
  | cons b bs ih =>
    intro h
    have hmem : b ∈ b :: bs := by simp
    have hab : a < b := h b hmem
    have hbs : ∀ x ∈ bs, a < x := fun x hx => h x (by simp [hx])
    have ihb : countLE a bs = 0 := ih hbs
    show (if b ≤ a then 1 else 0) + countLE a bs = 0
    split
    · next hba => exact absurd hab (by omega)
    · next _ => rw [ihb]

theorem orderStat_mem : ∀ (s : List Int) (k : Nat),
    1 ≤ k → k ≤ s.length → orderStat s k ∈ s := by
  intro s
  induction s with
  | nil =>
    intro k hk1 hkl
    simp at hkl
    exact absurd hk1 (by omega)
  | cons a as ih =>
    intro k hk1 hkl
    match k with
    | 0 => exact absurd hk1 (by omega)
    | 1 =>
      show a ∈ a :: as
      simp
    | Nat.succ (Nat.succ k') =>
      have hlen : k' + 1 ≤ as.length := by simp at hkl; omega
      have ihk := ih (k' + 1) (by omega) hlen
      show orderStat as (k' + 1) ∈ a :: as
      exact List.mem_cons_of_mem _ ihk

theorem countLE_orderStat :
    ∀ (s : List Int) (k : Nat), SortedScore s → 1 ≤ k → k ≤ s.length →
      countLE (orderStat s k) s = k := by
  intro s
  induction s with
  | nil =>
    intro k _ hk1 hkl
    simp at hkl
    exact absurd hk1 (by omega)
  | cons a as ih =>
    intro k hsort hk1 hkl
    match k with
    | 0 => exact absurd hk1 (by omega)
    | 1 =>
      have hzero : countLE a as = 0 := countLE_head_zero hsort.1
      show (if a ≤ a then 1 else 0) + countLE a as = 1
      split
      · next _ => rw [hzero]
      · next hbad => exact absurd (by omega : a ≤ a) hbad
    | Nat.succ (Nat.succ k') =>
      have htail : SortedScore as := hsort.2
      have hlen : k' + 1 ≤ as.length := by simp at hkl; omega
      have ihk : countLE (orderStat as (k' + 1)) as = k' + 1 :=
        ih (k' + 1) htail (by omega) hlen
      have hmem : orderStat as (k' + 1) ∈ as :=
        orderStat_mem as (k' + 1) (by omega) hlen
      have hlt : a < orderStat as (k' + 1) := hsort.1 _ hmem
      show (if a ≤ orderStat as (k' + 1) then 1 else 0)
          + countLE (orderStat as (k' + 1)) as = k' + 2
      split
      · next _ =>
        rw [ihk]
        omega
      · next hbad => exact absurd hlt (by omega)


theorem covered_of_orderStat_le :
    ∀ (s : List Int) (k : Nat) (q : Int), SortedScore s → 1 ≤ k → k ≤ s.length →
      orderStat s k ≤ q → k ≤ countLE q s := by
  intro s
  induction s with
  | nil =>
    intro k q _ hk1 hkl _
    simp at hkl
    exact absurd hk1 (by omega)
  | cons a as ih =>
    intro k q hsort hk1 hkl hq
    match k with
    | 0 => exact absurd hk1 (by omega)
    | 1 =>
      have ha : a ≤ q := hq
      show 1 ≤ (if a ≤ q then 1 else 0) + countLE q as
      split
      · next _ => omega
      · next hbad => exact absurd ha hbad
    | Nat.succ (Nat.succ k') =>
      have htail : SortedScore as := hsort.2
      have hlen : k' + 1 ≤ as.length := by simp at hkl; omega
      have hq' : orderStat as (k' + 1) ≤ q := hq
      have hmem : orderStat as (k' + 1) ∈ as :=
        orderStat_mem as (k' + 1) (by omega) hlen
      have hlt : a < orderStat as (k' + 1) := hsort.1 _ hmem
      have ha : a ≤ q := by omega
      have ihk : k' + 1 ≤ countLE q as := ih (k' + 1) q htail (by omega) hlen hq'
      show k' + 2 ≤ (if a ≤ q then 1 else 0) + countLE q as
      split
      · next _ => omega
      · next hbad => exact absurd ha hbad


theorem shift_minimality :
    ∀ (s : List Int) (km : Nat) (q : Int), SortedScore s → 1 ≤ km → km ≤ s.length →
      km ≤ countLE q s → orderStat s km ≤ q := by
  intro s
  induction s with
  | nil =>
    intro km q _ hk1 hkl _
    simp at hkl
    exact absurd hk1 (by omega)
  | cons a as ih =>
    intro km q hsort hk1 hkl hq
    match km with
    | 0 => exact absurd hk1 (by omega)
    | 1 =>
      by_cases ha : a ≤ q
      · show orderStat (a :: as) 1 ≤ q
        exact ha
      · have hz : countLE q as = 0 := countLE_head_zero (fun b hb => by
          have hlt := hsort.1 b hb
          omega)
        have hcount : countLE q (a :: as) = 0 := by
          show (if a ≤ q then 1 else 0) + countLE q as = 0
          split
          · next hpos => exact absurd hpos ha
          · next _ => rw [hz]
        omega
    | Nat.succ (Nat.succ km') =>
      have htail : SortedScore as := hsort.2
      have hlen : km' + 1 ≤ as.length := by simp at hkl; omega
      have hq' : km' + 2 ≤ (if a ≤ q then 1 else 0) + countLE q as := hq
      have ihk : orderStat as (km' + 1) ≤ q := by
        by_cases ha : a ≤ q
        · have hif : (if a ≤ q then 1 else 0) = 1 := by
            split
            · next _ => rfl
            · next hbad => exact absurd ha hbad
          rw [hif] at hq'
          apply ih (km' + 1) q htail (by omega) hlen
          omega
        · have hz : countLE q as = 0 := countLE_head_zero (fun b hb => by
            have hlt := hsort.1 b hb
            omega)
          have hcount : (if a ≤ q then 1 else 0) + countLE q as = 0 := by
            split
            · next hpos => exact absurd hpos ha
            · next _ => rw [hz]
          omega
      show orderStat (a :: as) (km' + 2) ≤ q
      exact ihk


def witness1 : List (Bool × Bool) := [(true, false), (false, false)]
def witness2 : List (Bool × Bool) := [(false, false), (false, false)]

theorem nonidentifiability_witness :
    goodR witness1 = goodR witness2 ∧
    goodL witness1 = 1 ∧ goodL witness2 = 0 ∧
    diffP witness1 ≤ 1 ∧ diffP witness2 = 0 := by
  decide

end PaperA
