title: Disclaimer
class: big

This is not a talk about parallelism, is about concurrency.

Parallelism:

- Program does same thing regardless of the amount of cores
- Used to speed up pure (non-IO monad) Haskell code

Concurrency:

- IO Monad everywhere!
- Effects from multiple threads are interleaved (non-deterministic)
- Necessary to deal with multiple sources of input/output

---
class: big image

<article class="flexbox vcenter">
  <img src="images/con_and_par.jpg" width="500px" height="400px" alt="Desktop Pic" title="Desktop Pic">
  <footer class="source">Courtesy of Joe Armstrong</footer>
</article>


---

title: Agenda
class: big
build_lists: true

Things we'll cover:


* Basic Concurrency:
    - forkIO
    - MVars


* Async Exceptions:
    - cancellation
    - timeout


* Software Transaction Memory (STM)

---

title: Threading
subtitle: using forkIO
class: segue nobackground dark

---

title: Forking threads

<pre class="prettyprint lang-hs bigger" data-lang="HASKELL">
forkIO :: IO () -> IO ThreadId
</pre>

- Creates a new thread to run on the IO action


- new thread runs "at the same time" as the current thread.


<aside class="note">

- dalap single traversal with custom tree modification at any level of the tree
- kibit uses backtracking to accomplish this
- You may both, kibit on your dalap transformation functions

</aside>

---

title: Interleaving example

<pre class="prettyprint lang-hs" data-lang="HASKELL">
import Control.Concurrent
import Control.Monad
import System.IO

main = do
  hSetBuffering stdout NoBuffering
  -- forkIO :: IO () -> IO ThreadId
  forkIO $ forever (putChar 'A')
  forkIO $ forever (putChar 'B')
  threadDelay (10 ^ 6)
</pre>


<pre class="" data-lang="BASH">
$ ghc fork.hs
[1 of 1] Compiling Main ( fork.hs, fork.o ) Linking fork ...

$ ./fork | tail -c 159
AAAAAAAAABABABABABABABABABABABABABABABABABABABABABABAB
ABABABABABABABABABABABABABABABABABABABABABABABABABABAB
ABABABABABABABABABABABABABABABABABABABABABABABABABABAB
</pre>

---

title: A note about performance
subtitle: threading is freaking cheap!

1 MILLION threads require around 1.5 GB of storage (approx. 1.5k per thread)

<center>
<img src="/images/use_the_forkio.jpeg" class="reflect" alt="Use the forkIO!">
</center>

---

title: Communication
subtitle: using MVars
class: segue nobackground dark

---

title: MVar
subtitle: The most basic concurrency tool in Haskell


<pre class="prettyprint lang-hs" data-lang="HASKELL">
data MVar a -- Abstract Constructor

newEmptyMVar :: IO (MVar a)
takeMVar     :: MVar a -> IO a
putMVar      :: MVar a -> a -> IO ()
</pre>

- Wraps a value (Image a wrapper class in OO land)
- It has 2 states: <b>empty</b> or <b>full</b>
- When full and want to put a value, <b>blocks</b>
- When empty and want to take a value, <b>blocks</b>
- Doesn't block otherwise

---

title: Downloading URLs concurrently
subtitle: First attempt

<pre class="prettyprint lang-hs" data-lang="HASKELL">
getURL :: String -> IO String

downloadPages = do
  m1 <- newEmptyMVar
  m2 <- newEmptyMVar

  forkIO $ getURL "http://www.wikipedia.org/wiki/haskell" >>= putMVar m1
  forkIO $ getURL "http://www.haskell.org/" >>= putMVar m2

  r1 <- takeMVar m1
  r2 <- takeMVar m2

  return (r1, r2)
</pre>

---

title: Downloading URLs concurrently
subtitle: Abstracting the pattern

<pre class="prettyprint lang-hs" data-lang="HASKELL">

newtype Async a = Async (MVar a)

async :: IO a -> IO (Async a)
async action = do
  m <- newEmptyMVar
  forkIO $ action >>= putMVar m
  return (Async m)

wait :: Async a -> IO a
wait (Async m) = readMVar m
-- ^ read != take, doesn't take out the value from
-- the MVar box

</pre>

---

title: Downloading URLs concurrently
subtitle: Second Attempt

<pre class="prettyprint lang-hs" data-lang="HASKELL">

downloadPages = do

  a1 <- async $ getURL "http://www.wikipedia.org/wiki/haskell"
  a2 <- async $ getURL "http://www.haskell.org/"
  r1 <- wait a1
  r2 <- wait a2

  return (r1, r2)

</pre>

---

title: Downloading URLs concurrently
subtitle: Third Attempt

<pre class="prettyprint lang-hs" data-lang="HASKELL">
sites = [ "http://www.google.com"
        , "http://www.bing.com"
        ...
        ]

downloadPages = mapM (async . http) sites >>= mapM wait
  where
    http url = do
      (result, time) <- timeit $ getURL url
      printf "downloaded: %s (%d bytes, %.2fs)\n" url (length result) time
</pre>

<pre class="" data-lang="OUTPUT">
downloaded: http://www.google.com (14524 bytes, 0.17s)
downloaded: http://www.bing.com (24740 bytes, 0.18s)
downloaded: http://www.yahoo.com (153065 bytes, 1.11s)
</pre>

---

title: An MVar could be:

- <b>lock</b>
    - \``MVar ()`\` behaves like a lock: full is unlocked, empty is locked
    - Can be used as a mutex to protect some other shared state

- <b>one-spot-channel</b>
    - Since an MVar holds at most one value, it behaves like an
      async channel with buffer-size of one

- <b>building block</b>
    - MVar can be used to build many different concurrent data
      structures/abstractions

---

title: A note on fairness in MVars

- Threads blocked on an MVar are processed in a FIFO fashion

- Each `putMVar` wakes exactly <b>one</b> thread, and performs
  blocked operation atomically

- Fairness can lead to alternation when two threads compete for an MVar
    - thread A: takeMVar (succeeds)
    - thread B: takeMVar (blocks)
    - thread A: putMVar (succeeds, and wakes thread B)
    - thread A: takeMVar again (blocks)
    - Cannot break cycle until a thread is pre-empted (re-scheduled)
      while MVar is full

- MVar contention may be expensive!

---

title: Unbound Channels

<pre class="prettyprint lang-hs" data-lang="HASKELL">
data Chan a

newChan :: IO (Chan a)
writeChan :: Chan a -> a -> IO ()
readChan  :: Chan a -> IO a
</pre>

- Writes to the Chan don't block
- Reads to Chan don't block until Chan is empty
- Great to use as a task queue on a producer/consumer scenario
- Implemented using a list-like of MVars internally

---

title: Interruption / Cancelation
subtitle: asynchrounous exceptions
class: segue nobackground dark

---

title: Motivations

- There are many cases when we want to interrupt a thread:
    - Web browser stop button
    - Timeout on slow connections
    - Cancel an operation that is pending, but not needed any more
    - etc.

---

title: Interrupting a thread
subtitle: introducing `throwTo`

<pre class="prettyprint lang-hs" data-lang="HASKELL">
throwTo :: Exception e => ThreadId -> e -> IO ()
</pre>

- Throws an async exception `e` on the thread pointed by the `ThreadId`
- Interruption appears as an exception
    - Good: We need exception handling to clean up possible errors, the
      same handlers could be used for interruptions too.
    - Exception safe code will be fine with an interruption.

    <pre class="prettyprint lang-hs" data-lang="HASKELL">
      bracket (newTempFile "temp")       -- open
              (\file -> removeFile file) -- clenaup
              (\file -> ...)             -- do
    </pre>

---

title: Example
subtitle: Extending Async to handle interruptions

Let's add a cancel function to the Async type

<pre class="prettyprint lang-hs" data-lang="HASKELL">
newtype Async a = Async (MVar a)

async :: IO a -> IO (Async a)
async io = do
  m <- newEmptyMVar
  forkIO $ do r <- io; putMVar m r
  return (Async m)

wait :: Async a -> IO a
wait (Async m) = readMVar m

-- we want to add
cancel :: Async a -> IO ()
</pre>

---

title: Example
subtitle: Modify Async definition

<pre class="prettyprint lang-hs" data-lang="HASKELL">
newtype Async a = Async <b>ThreadId</b> (MVar a)

async :: IO a -> IO (Async a)
async io = do
  m <- newEmptyMVar
  <b>tid</b> <- forkIO $ do r <- io; putMVar m r
  return (Async <b>tid</b> m)
<b>
cancel :: Async a -> IO ()
cancel (Async tid _) = throwTo tid ThreadKilled
</b>
</pre>


---

title: Example
subtitle: Modify Async definition

But what about <b>wait</b>? Previously it had type

<pre class="prettyprint lang-hs" data-lang="HASKELL">
  wait :: Async a -> IO ()
</pre>

Should it return if the `Async` was cancelled?

Cancellation is an exception, so wait should return the Exception
that was thrown...

<b>Extra WIN</b>: safe handling of other errors as well

---

title: Example
subtitle: Async with exception support

<pre class="prettyprint lang-hs" data-lang="HASKELL">

newtype Async a = Async ThreadId <b>(MVar (Either SomeException a))</b>

async :: IO a -> IO (Async a)
async io = do
  m <- newEmptyMVar
  tid <- forkIO $ <b>returnRight `catch` returnLeft</b>
where
  <b>returnRight = do
    r <- io
    putMVar m (Right r)
  returnLeft (e :: SomeException) =
    putMVar m (Left e)</b>

<b>wait :: Async a -> IO (Either SomeException a)</b>
wait (Async _ var) = readMVar var

</pre>

---

title: Example
subtitle: Using Async with cancellation

<pre class="prettyprint lang-hs" data-lang="HASKELL">

main = do
  asyncs <- mapM (async . http) sites

  forkIO $ do
    hSetBuffering stdin NoBuffering
    forever $ do
      c <- getChar
      -- hit q stops downloads
      <b>when (c == 'q') $ mapM_ cancel asyncs</b>

  rs <- mapM wait asyncs
  -- ^ blocks until all async actions are done / cancelled

  printf "%d/%d finished\n" (length (rights rs)) (length rs)

</pre>

---

title: Example
subtitle: Output
build_lists: true

<pre class="" data-lang="BASH">
./geturlscancel
downloaded: http://www.google.com (14538 bytes, 0.17s)
downloaded: http://www.bing.com (24740 bytes, 0.22s)
q2/5 Finished
</pre>

Points to note:

- We are using a complicated HTTP library underneath, yet it
  supports interruption automatically

- Having async interruption be the default is powerful

- Not a silver bullet: With truly mutable state, interruptions
  can be difficult.

- STATE PROPAGATED EVERYWHERE == COMPLEXITY


---

title: Software Transactional Memory
class: segue nobackground dark

---

title: STM
subtitle: What is it?

- An alternative to MVar for managing
    - shared state
    - communication

- STM has several advantages:
    - compositional (Monads FTW)
    - much easier to get right (no deadlocks)
    - much easier to manage error conditions (async exceptions included)

---

title: Example
subtitle: A Window Manager

<article class="flexbox vcenter">
  <img src="images/Desktop.png" width="500px" height="400px" alt="Desktop Pic" title="Desktop Pic">
</article>

---

title: Window Manager
subtitle: Implementation details

Suppose we want to have one thread for each input/output stream:

- On thread to listen to the user
- One thread for each client application
- One thread to render the display
- All threads share the state of the desktops, at the same time.

How should we represent this using Haskell's toolbelt?

---

title: Window Manager
subtitle: Option 1: a single MVar for the whole state

<pre class="prettyprint lang-hs" data-lang="HASKELL">
type Display = MVar (Map Desktop (Set Window))
</pre>
Advantages:

- Simple

Disadvantages:

- Single point of contention.
    - Missbehaving thread can block everyone else
    - Performance penalties

---

title: Window Manager
subtitle: Option 2: an MVar per Desktop

<pre class="prettyprint lang-hs" data-lang="HASKELL">
<strike>type Display = MVar (Map Desktop (Set Window))</strike>
type Display = Map Desktop (MVar (Set Window))
</pre>

Avoids single point of contention, but new problem emerges:

<pre class="prettyprint lang-hs" data-lang="HASKELL">
moveWindow :: Display -> Window -> Desktop -> Desktop -> IO ()
moveWindow disp win desktopA desktopB = do
    <b>windowSetA <- takeMVar mWindowSetA
    -- other threads say hello to inconsistent intermmidate state
    windowSetB <- takeMVar mWindowSetB</b>
    putMVar mWindowSetA (Set.delete win windowSetA)
    putMVar mWindowSetB (Set.insert win windowSetB)
  where
    mWindowSetA = fromJust (Map.lookup desktopA disp)
    mWindowSetB = fromJust (Map.lookup desktopB disp)
</pre>

---

title: Window Manager
subtitle: Dinning Philosophers
build_lists: true

<pre class="prettyprint lang-hs" data-lang="HASKELL">
moveWindow disp win desktopA desktopB = do
    <b>windowSetA <- takeMVar mWindowSetA
    windowSetB <- takeMVar mWindowSetB</b>
    ...
</pre>

- Thread 1 (T1): calls `moveWindow disp w1 a b`
- Thread 2 (T2): calls `moveWindow disp w2 b a`
- T1 takes `MVar` of `Desktop a`
- T2 takes `MVar` of `Desktop b`
- T1 tries to take `MVar` for `Desktop b`, blocks...
- T2 tries to take `MVar` for `Desktop a`, blocks...
- <b>DEADLOCK</b>

---

title: Window Manager
subtitle: Can we solve this with MVars?
build_lists: true

We could, but requires a high price:

- Impose fixed ordering on `MVars`, make `takeMVar` calls in the same order in <b>every</b> thread.
    - Libraries must obey this rules
    - Error-Checking can be done at runtime, complicated...
    - <img src="images/fuckthat.jpg"></img>

---

title: Window Manager
subtitle: STM to the rescue

<pre class="prettyprint lang-hs" data-lang="HASKELL">
type Display = Map Desktop <b>(TVar (Set Window))</b>

moveWindow :: Display -> Window -> Desktop -> Desktop -> IO ()
moveWindow disp win a b = <b>atomically</b> $ do
  wa <- readTVar ma
  wb <- readTVar mb
  writeTVar ma (Set.delete win wa)
  writeTVar mb (Set.insert win wb)
where
  ma = fromJust (Map.lookup a disp)
  mb = fromJust (Map.lookup a disp)
</pre>

- Operations inside `atomically` happen indivisibly to the rest of the program (transaction)
- Ordering is irrelevant - we can interleave read/write actions

---

title: STM
subtitle: Basic API

<pre class="prettyprint lang-hs" data-lang="HASKELL">

data STM a -- abstract
instance Monad STM -- amongst other things
atomically :: STM a -> IO a

data TVar a -- abstract
newTVar   :: STM (TVar a)
readTVar  :: TVar a -> STM a
writeTVar :: TVar a -> a -> STM ()

</pre>

Implementation doesn't use a global lock, two transactions operating on
disjoint sets of TVars can work simultaneously.

---

title: STM
subtitle: Composability

Write an operation to swap to Windows

<pre class="prettyprint lang-hs" data-lang="HASKELL">
swapWindows :: Display
            -> Window -> Desktop
            -> Window -> Desktop
            -> IO ()
</pre>

With `MVars` we would have to write a special purpose routine, on STM on the other hand

<pre class="prettyprint lang-hs" data-lang="HASKELL">
swapWindows disp w a v b = <b>atomically</b> $ do
  moveWindowsSTM disp w a b
  moveWindowsSTM disp v b a
  -- moveWindows fn seen previously on the STM monad
</pre>

STM allows composition of stateful operations into larger transactions

---

title: STM
subtitle: Blocking

Concurrent algorithms often times need a way to block execution
to wait for some condition to comply

<pre class="prettyprint lang-hs" data-lang="HASKELL">
retry :: STM a
</pre>

Semantics of retry is "try current transaction again", when a TVar in the transaction
changes (no busy loop)

<pre class="prettyprint lang-hs" data-lang="HASKELL">
atomically $ do
  x <- readTVar v
  if x == 0
    then retry
    else return x
</pre>

This thread will resume when some other thread puts in the `TVar v`
a value that is not 0.

---

title: STM
subtitle: Benefits and Woes

- Composable atomicity
- Composable blocking
- Robustness: easy error handling
- Slow on very large transactions
- Why would you use MVar?
    - fairness
    - single wakeup
    - performance
