"""AgileX Piper (6-DoF + gripper) support: FK, dataset transform, PolicyIK glue.

Used for the ACT -> EE-pose -> PolicyIK pipeline on the Piper arm. The 6 arm
joints (joint1-6) are FK'd to a TCP site added on link6 of the Menagerie model
(`models/piper/piper.xml`); the gripper is passed through.
"""
